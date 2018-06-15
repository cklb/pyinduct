"""
New simulation module approach that makes use of sympy for expression handling
"""

from copy import copy
import sympy as sp
# import symengine as sp
from time import clock
from sympy.utilities.lambdify import implemented_function
from sympy.functions.special.polynomials import jacobi
import numpy as np
# from tqdm import tqdm

from .registry import get_base
from .core import Domain, Function

time = sp.symbols("t", real=True)
space = sp.symbols("z:3", real=True)


# parameter database
_parameters = {}


def register_parameters(*args, **kwargs):
    if args:
        new_params = dict(args)
        _parameters.update(new_params)

    if kwargs:
        _parameters.update(kwargs)


def get_parameters(*args):
    if not args:
        return _parameters.items()
    else:
        return [_parameters.get(arg, None) for arg in args]


_weight_cnt = 0
_weight_letter = "_c"


def get_weight():
    global _weight_cnt
    w = sp.Function("{}_{}".format(_weight_letter, _weight_cnt),
                    real=True)(time)
    _weight_cnt += 1
    return w


def get_weights(num):
    if num < 0:
        raise ValueError
    c = [get_weight() for i in range(num)]
    return c


_function_cnt = 0
_function_letter = "_f"


def get_function(sym):
    global _function_cnt
    f = sp.Function("{}_{}".format(_function_letter, _function_cnt),
                    real=True)(sym)
    _function_cnt += 1
    return f


_field_var_cnt = 0
_field_var_letter = "_x"


def get_field_variable(*symbols):
    global _field_var_cnt
    x = sp.Function("{}_{}".format(_field_var_letter, _field_var_cnt),
                    real=True)(*symbols)
    _field_var_cnt += 1
    return x


_test_function_cnt = 0
_test_function_letter = "_g"


def get_test_function(*symbols):
    global _test_function_cnt
    g = sp.Function("{}_{}".format(_test_function_letter, _test_function_cnt),
                    real=True)(*symbols)
    _test_function_cnt += 1
    return g


_input_cnt = 0
_input_letter = "_u"


def get_input():
    global _input_cnt
    u = sp.Function("{}_{}".format(_input_letter, _input_cnt),
                    real=True)(time)
    _input_cnt += 1
    return u


class LumpedApproximation:

    def __init__(self, expr, weights, base_map, bcs):
        self._expr = expr
        self._weights = weights
        self._base_map = base_map
        self._bcs = bcs

        # substitute all known functions and symbols and generate callback
        impl_base = [(key, implemented_function(key.func, val)(*key.args))
                     for key, val in base_map.items()]
        impl_expr = expr.subs(impl_base)
        self._cb = sp.lambdify(weights,
                               [expr.subs(get_parameters())],
                               modules="numpy")

    @property
    def expression(self):
        return self._expr

    @property
    def weights(self):
        return self._weights

    @property
    def base_map(self):
        return self._base_map

    @property
    def base(self):
        """ Return all base fractions that do not coincide with essential bcs
        """
        return [frac for frac, ess in self._base_map.values() if not ess]

    @property
    def bcs(self):
        return self._bcs

    def __call__(self, *args, **kwargs):
        if args:
            return self._cb(args)
        if kwargs:
            return self._cb([kwargs[w] for w in self._weights])


class InnerProduct(sp.Expr):
    """ An unevaluated Inner Product on L2

    Args:
        a: Left Operand .
        b: Right Operand .

    """
    is_complex = True

    def __new__(cls, left, right, bounds):
        # identify free variables of arguments
        variables = left.atoms(sp.Symbol).union(right.atoms(sp.Symbol))

        # extract spatial coordinates
        variables = variables.intersection(space)

        # construct integrals over remaining coordinates
        # TODO .coeff seems to have problems with conjugate
        kernel = left * right
        # kernel = sp.conjugate(left) * right
        for dim in variables:
            xab = (dim, bounds[0], bounds[1])
            kernel = FakeIntegral(kernel, xab)
        return kernel

    @property
    def left(self):
        return self.args[0]

    @property
    def right(self):
        return self.args[1]

    def doit(self, **hints):
        pass


class FakeIntegral(sp.Integral):
    """
    Placeholder class that looks like an Integral but won't try to integrate
    anything.

    Instances of this object will later be used to perform numerical integration
    where possible.
    """

    def doit(self, **hints):
        """
        Do not try to integrate anything, just perform operations on the args
        """
        return self.func(*[arg.doit() for arg in self.args])


class Lagrange1stOrder(sp.Piecewise):

    def __new__(cls, start, mid, end):
        x = sp.Symbol("x")
        obj = sp.Piecewise.__new__(cls, (0, x < start),
                                   ((x-start)/(mid - start), x < mid),
                                   (1, x == mid),
                                   (1 - (x-mid)/(end - mid), x < end),
                                   (0, x >= end),
                                   )
        return obj


def build_lag1st(sym, start, mid, end):
    if start == mid:
        obj = sp.Piecewise((0, sym < mid),
                           (1 - (sym - mid)/(end - mid), sym <= end),
                           (0, sym > end)
                           )
    elif mid == end:
        obj = sp.Piecewise((0, sym < start),
                           ((sym - start)/(mid - start), sym <= mid),
                           (0, sym > mid)
                           )
    else:
        obj = sp.Piecewise((0, sym < start),
                           ((sym - start)/(mid - start), sym <= mid),
                           (1 - (sym - mid)/(end - mid), sym <= end),
                           (0, sym > end)
                           )
    return obj


def create_lag1ast_base(sym, bounds, num):
    # define base vectors
    nodes = Domain(bounds, num)
    funcs = [build_lag1st(sym, *nodes[[0, 0, 1]])]
    for idx in range(1, num-1):
        funcs.append(build_lag1st(sym, *nodes[[idx-1, idx, idx+1]]))
    funcs += [build_lag1st(sym, *nodes[[-2, -1, -1]])]

    if 0:
        # plotting of Piecewise fails really bad, so lambdify first
        fig = plt.figure()
        vals = np.linspace(0, 1, num=1e3)
        for f in funcs:
            fn = sp.lambdify(z, f, modules="numpy")
            res = fn(vals)
            plt.plot(vals, res)
        plt.show()

    return funcs


def create_approximation(sym, base_lbl, boundary_conditions, weights=None):
    """
    Create a lumped approximation of a distributed variable
    """
    base = get_base(base_lbl)
    ess_bcs, nat_bcs, boundary_positions = _classify_boundaries(
        sym,
        boundary_conditions)

    ess_pairs = []
    ess_idxs = []
    # find the corresponding functions for the essential boundaries
    for cond, pos in ess_bcs:
        # identify all base fractions that differ from zero
        for idx, func in enumerate(base):
            if isinstance(func, Function):
                res = func(pos)
            elif isinstance(func, sp.Basic):
                res = func.subs(sym, pos)
            else:
                raise NotImplementedError

            if res != 0:
                ess_pairs.append((cond, func))
                ess_idxs.append(idx)

        if len(ess_pairs) < len(ess_bcs):
            # no suitable base fraction for homogenisation found, create one
            inhom_pos = pos
            hom_pos = next((p for p in boundary_positions if p != pos))
            hom_frac = create_hom_func(sym, inhom_pos, hom_pos)
            ess_pairs.append((cond, hom_frac))

        assert len(ess_pairs) == len(ess_bcs)

    base_mapping = {}

    # inhomogeneous parts that are to be excluded
    x_ess = 0
    for cond, func in ess_pairs:
        d_func = get_function(sym)
        x_ess += cond[1] * d_func
        base_mapping[d_func] = func, True

    # extract shape functions which are to be kept
    nat_funcs = [func for idx, func in enumerate(base) if idx not in ess_idxs]

    # homogeneous part
    x_nat = 0
    if weights is None:
        weights = get_weights(len(nat_funcs))
    for idx, func in enumerate(nat_funcs):
        d_func = get_function(sym)
        x_nat += weights[idx] * d_func
        base_mapping[d_func] = func, False

    x_comp = x_ess + x_nat
    x_approx = LumpedApproximation(x_comp, weights, base_mapping, boundary_conditions)

    return x_approx


def _classify_boundaries(sym, boundary_conditions):
    """ Identify essential and natural boundaries
    """
    boundary_positions = []
    ess_bcs = []
    nat_bcs = []
    for cond in boundary_conditions:
        lhs_idx = next((idx for idx, arg in enumerate(cond.args)
                        if _field_var_letter in str(arg.atoms(sp.Function))))
        lhs = cond.args[lhs_idx]
        rhs = cond.args[1 - lhs_idx]
        if isinstance(lhs, sp.Subs):
            assert sym in lhs.args[1]
            assert len(lhs.args[2]) == 1
            pos = next(iter(lhs.args[2]))
            expr = lhs.args[0]
        else:
            # go f*ck yourself symengine
            raise NotImplementedError

        pair = ((lhs, rhs), pos)

        if isinstance(expr, sp.Derivative):
            nat_bcs.append(pair)
        else:
            ess_bcs.append(pair)

        boundary_positions.append(pos)


    return ess_bcs, nat_bcs, boundary_positions


def create_hom_func(sym, pos_a, pos_b, mode="linear"):
    r"""
    Build a symbolic expression for homogenisation purposes.

    This function will evaluate to one at *pos_a* and to zero at *pos_b*.

    Args:
        sym: Symbol to use.
        pos_a: First position.
        pos_b: Second Position.

    Keyword Args:
        mode(str): Function type to use, choices are "linear", "trig" and "exp".

    Returns: Sympy expression

    """
    if mode == "linear":
        variables = sp.symbols("_m _n")
        _f = variables[0]*sym + variables[1]
    elif mode == "trig":
        variables = sp.symbols("_a _b")
        _f = sp.cos(variables[0] * sym + variables[1])
    # elif mode == "exp":
    #     variables = sp.symbols("_c _b")
    #     _f = sp.cos(variables[0]*sym + variables[1])
    else:
        raise NotImplementedError

    eqs = [_f.subs(pos_a) - 1, _f.subs(pos_b - 0)]
    sol = sp.solve(eqs, variables, dict=True)[0]
    res = _f.subs(sol)

    return res


def substitute_approximations(weak_form, mapping):

    ext_form, ext_mapping = expand_kth_terms(weak_form, mapping)

    def wrap_expr(expr, *sym):
        def wrapped_func(*_sym):
            return expr.subs(list(zip(sym, _sym)))

        return wrapped_func

    def gen_func_subs_pair(func, expr):
        if isinstance(func, sp.Function):
            a = func.func
            args = func.args
        elif isinstance(func, sp.Subs):
            a = func
            if isinstance(func.args[0], sp.Derivative):
                args = func.args[0].args[0].args
            elif isinstance(func.args[0], sp.Function):
                args = func.args[0].args
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        b = wrap_expr(expr, *args)
        return a, b

    # create substitution list
    rep_list = []
    for sym, approx in ext_mapping.items():
        if isinstance(approx, LumpedApproximation):
            rep_list.append(gen_func_subs_pair(sym, approx.expression))
            rep_list += [gen_func_subs_pair(*bc.args) for bc in approx.bcs]
        else:
            rep_list.append(gen_func_subs_pair(sym, approx))

    # substitute formulations
    rep_eqs = []
    # for eq in tqdm(ext_form):
    for eq in ext_form:
        rep_eq = eq
        for pair in rep_list:
            rep_eq = rep_eq.replace(*pair, simultaneous=True)

        rep_eqs.append(rep_eq.doit())

    return rep_eqs


def expand_kth_terms(weak_form, mapping):
    """ Search for kth terms and expand the equation system by their mappings"""

    new_exprs = []
    new_mapping = {}
    kth_placeholders = {}
    # search for kth placeholders (by now given by testfunctions)
    for sym, approx in mapping.items():
        if _test_function_letter in str(sym):
            kth_placeholders[sym] = approx
        else:
            new_mapping[sym] = approx

    # replace the kth occurrences by adding extra equations for every entry
    for eq in weak_form:
        occurrences = [sym in eq.atoms(sp.Function)
                       for sym in kth_placeholders.keys()]
        if not any(occurrences):
            # if no placeholder is used, directly forward the equation
            new_exprs.append(eq)
        else:
            for sym, approx in kth_placeholders.items():
                new_eqs, new_map = _substitute_kth_occurrence(eq, sym, approx)
                new_exprs += new_eqs
                new_mapping.update(new_map)

    return new_exprs, new_mapping


def _substitute_kth_occurrence(equation, symbol, expressions):
    new_eqs = []
    mappings = {}
    for expr in expressions:
        if equation.atoms(symbol):
            _f = get_test_function(*symbol.args)
            new_eqs.append(equation.replace(symbol, _f))
            mappings[_f] = expr

    return new_eqs, mappings


def create_first_order_system(weak_forms):
    temp_derivatives = _find_derivatives(weak_forms, time)

    new_forms = weak_forms[:]
    for der in temp_derivatives:
        new_forms, subs_list, new_targets = _convert_higher_derivative(new_forms, der)

    # replace functions with dummies
    for der in temp_derivatives:
        new_forms, mapping, targets = _substitute_temporal_derivative(new_forms, der)
        new_mapping.update(mapping)
        new_targets.update(targets)

    print(new_targets)
    print(new_mapping)
    # solve for targets
    # sol_dict = sp.solve(new_forms, targets)

    return new_forms


def _find_derivatives(weak_form, sym):
    """ Find derivatives """
    temp_derivatives = set()
    for eq in weak_form:
        t_ders = [der for der in eq.atoms(sp.Derivative) if der.args[1] == sym]
        # pairs = [()]
        temp_derivatives.update(t_ders)

    return temp_derivatives


def _convert_higher_derivative(weak_forms, derivative):

    target_set = set()
    subs_list = []

    expr = derivative.args[0]
    order = len(derivative.args) - 1

    if _weight_letter in str(expr):
        if order > 1:
            # c1_dd = ...
            # c1_d = c2
            # -> c1_dd = c2_d
            # -> c2_d = ...
            new_var = get_weight()
            new_deriv = derivative.func(*derivative.args[:-1])
            new_eq = new_var - new_deriv
            new_forms = [form.subs(derivative, new_var.diff(time))
                         for form in weak_forms]
            new_forms.append(new_eq)
            if order > 2:
                new_forms, subs_list, target_set = _convert_higher_derivative(
                    new_forms, new_deriv)
        else:
            d_der = sp.Dummy()
            subs_pair = derivative, d_der
            subs_list.append(subs_pair)
            target_set.add(d_der)
    elif _input_letter in str(expr):
        # u1_d = ...
        # v = u1_d
        # -> v_d = u1_dd
        # -> c2_d = ...
        new_var = get_weight()
        new_deriv = derivative.func(*derivative.args[:-1])
        new_eq = new_var - new_deriv
        new_forms = [form.subs(derivative, new_var.diff(time))
                     for form in weak_forms]
        new_forms.append(new_eq)
        if order > 2:
            new_forms, subs_list, target_set = _convert_higher_derivative(
                new_forms, new_deriv)

    return new_forms, subs_list, target_set


def _substitute_temporal_derivative(weak_forms, derivative):

    subs_list = []
    target_set = set()

    expr = derivative.args[0]
    order = len(derivative.args) - 1

    d_der = sp.Dummy()
    if _weight_letter in str(expr) and order > 1:
        # c1_dd = ...
        # c1_d = c2
        # -> c1_dd = c2_d
        # -> c2_d = ...
        new_var = get_weight()
        new_deriv = derivative.func(*derivative.args[:-1])
        new_eq = new_var - new_deriv
        weak_forms.append(new_eq)
        subs_pair = (derivative, new_var.diff(t))
        weak_forms, new_s_list, new_t_set = _substitute_temporal_derivative(
            weak_forms,
            new_deriv)
        subs_list += new_s_list
        target_set.update(new_t_set)

    if _input_letter in str(expr) and order > 0:
        pass

    if _weight_letter in str(expr) and order == 1:
        target_set.add(d_der)
    subs_list.append(subs_pair)
    new_forms = [form.subs(subs_list) for form in weak_forms]

    return new_forms, subs_list, target_set
