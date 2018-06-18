"""
New simulation module approach that makes use of sympy for expression handling
"""

from copy import copy
import sympy as sp
# import symengine as sp
from time import clock
from scipy.integrate import quad
# from mpmath import quad
# from sympy.mpmath import quad
from sympy.utilities.lambdify import implemented_function
from sympy.functions.special.polynomials import jacobi
import numpy as np
# from tqdm import tqdm

from matplotlib import pyplot as plt

from .registry import get_base
from .core import Domain, Function, domain_intersection, integrate_function

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


_lambda_cnt = 0
_input_letter = "_u"


def get_input():
    global _lambda_cnt
    u = sp.Function("{}_{}".format(_input_letter, _lambda_cnt),
                    real=True)(time)
    _lambda_cnt += 1
    return u


# _lambda_cnt = 0
# _lambda_letter = "_l"


# def get_lambda():
#     global _lambda_cnt
#     u = sp.Function("{}_{}".format(_lambda_letter, _lambda_cnt),
#                     real=True)(time)
#     _lambda_cnt += 1
#     return u


class LumpedApproximation:

    def __init__(self, expr, weights, base_map, bcs):
        self._expr = expr
        self._weights = weights
        self._base_map = base_map
        self._bcs = bcs

        # substitute all known functions and symbols and generate callback
        # impl_base = [(key, implemented_function(key.func, val)(*key.args))
        #              for key, val in base_map.items()]
        # impl_expr = expr.subs(impl_base)
        # self._cb = sp.lambdify(weights,
        #                        [impl_expr.subs(get_parameters())],
        #                        modules="numpy")

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

    def eval_numerically(self):
        # build callback
        kernel, (sym, a, b) = self.args
        f = sp.lambdify(sym, kernel, modules="numpy")

        # extract domains
        domain = {(float(a), float(b))}
        nonzero = domain
        for func in kernel.atoms(sp.Function):
            if hasattr(func, "_imp_"):
                new_dom = func._imp_.domain
                domain = domain_intersection(domain, new_dom)
                nonzero = domain_intersection(nonzero, func._imp_.nonzero)

        interval = domain_intersection(domain, nonzero)

        # perform integration
        if interval:
            if 1:
                reg = next(iter(interval))
                x_vals = np.linspace(float(a), float(b))
                plt.plot(x_vals, f(x_vals).T)
                plt.axhline(xmin=reg[0], xmax=reg[1], c="r")
                plt.show()

            res, err = integrate_function(f, interval)
        else:
            res = 0
        return res


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
        d_func.func._imp_ = staticmethod(func)
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
        d_func.func._imp_ = staticmethod(func)
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

    # create substitution lists
    rep_list = []
    # subs_list = []
    for sym, approx in ext_mapping.items():
        if isinstance(approx, LumpedApproximation):
            rep_list.append(gen_func_subs_pair(sym, approx.expression))
            rep_list += [gen_func_subs_pair(*bc.args) for bc in approx.bcs]
            # subs_list += [implemented_function(key.func, func)(*key.args)
            #               for key, (func, flag) in approx.base_map.items()]
        else:
            rep_list.append(gen_func_subs_pair(sym, approx))

    # substitute symbolic approximations
    rep_eqs = []
    # for eq in tqdm(ext_form):
    for eq in ext_form:
        rep_eq = eq
        for pair in rep_list:
            rep_eq = rep_eq.replace(*pair, simultaneous=True)

        rep_eqs.append(rep_eq.doit())

    # substitute callbacks
    # impl_expr = expr.subs(impl_base)

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
            _g = get_test_function(*symbol.args)
            _g.func._imp_ = staticmethod(expr)
            new_eqs.append(equation.replace(symbol, _g))
            mappings[_g] = expr

    return new_eqs, mappings


def create_first_order_system(weak_forms):

    derivatives = _find_derivatives(weak_forms, time)
    derivatives.update(_find_derivatives(weak_forms, space))

    new_forms = weak_forms[:]
    subs_list = []
    targets = set()
    for der in derivatives:
        new_forms, subs_pairs, target_set = _convert_higher_derivative(
            new_forms, der)
        subs_list += subs_pairs
        targets.update(target_set)

    # sort targets
    def _find_entry(entry):
        return next((a for a, b in subs_list if b == entry))

    sorted_targets = sorted(targets, key=lambda x: str(_find_entry(x)))
    # print(targets)
    # print(subs_list)
    # print(new_forms)

    # substitute weights with dummies
    state_elems = set()
    for form in new_forms:
        funcs = form.atoms(sp.Function)
        for f in funcs:
            if _weight_letter in str(f):
                state_elems.add(f)
    sorted_state = sorted(state_elems, key=lambda x: str(x))
    state_mapping = [(state, sp.Dummy()) for state in sorted_state]
    new_forms = [form.subs(state_mapping) for form in new_forms]

    # simplify integrals
    new_forms = _simplify_integrals(new_forms)
    sp.pprint(new_forms)

    # solve integrals
    new_forms = _solve_integrals(new_forms)
    sp.pprint(new_forms)

    # solve for targets
    sol_dict = sp.solve(new_forms, sorted_targets)
    sp.pprint(sol_dict)

    # build statespace form
    ss_form = sp.Matrix([sol_dict[target] for target in sorted_targets])

    return ss_form, subs_list


def _find_integrals(weak_forms):
    """ Find fake integrals """
    integrals = set()
    for eq in weak_forms:
        ints = [_int for _int in eq.atoms(FakeIntegral)]
        integrals.update(ints)

    return integrals


def _find_derivatives(weak_forms, sym):
    """ Find derivatives """
    if isinstance(sym, sp.Symbol):
        sym = set([sym])

    derivatives = set()
    for eq in weak_forms:
        ders = {der for der in eq.atoms(sp.Derivative) if der.args[1] in sym}
        derivatives.update(ders)

    return derivatives


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
            red_deriv = derivative.func(*derivative.args[:-1])
            new_eq = new_var - red_deriv
            new_der = new_var.diff(time)
            new_forms = [form.subs(derivative, new_der)
                         for form in weak_forms]
            new_forms.append(new_eq)

            # replace remaining derivative
            new_forms, subs_pairs, new_targets = _convert_higher_derivative(
                new_forms, red_deriv)
            subs_list += subs_pairs
            target_set.update(new_targets)

            # replace newly introduced one
            new_forms, subs_pairs, new_targets = _convert_higher_derivative(
                new_forms, new_der)
            subs_list += subs_pairs
            target_set.update(new_targets)
        else:
            d_der = sp.Dummy()
            subs_pairs = [(derivative, d_der)]
            new_forms = [form.subs(subs_pairs) for form in weak_forms]
            subs_list += subs_pairs
            target_set.add(d_der)

    elif _input_letter in str(expr):
        raise NotImplementedError

    elif _function_letter in str(expr) or _test_function_letter in str(expr):
        # derive the associated callbacks
        if _function_letter in str(expr):
            d_func = get_function(*expr.args)

        elif _test_function_letter in str(expr):
            d_func = get_test_function(*expr)

        callback = expr.func._imp_
        d_func.func._imp_ = staticmethod(callback.derive(order))
        subs_pair = derivative, d_func
        new_forms = [form.replace(*subs_pair) for form in weak_forms]
        subs_list += [subs_pair]

    return new_forms, subs_list, target_set


def _simplify_integrals(weak_forms):

    integrals = _find_integrals(weak_forms)

    subs_list = []
    for integral in integrals:
        red_int = _reduce_kernel(integral)
        subs_list.append((integral, red_int))

    red_forms = [f.subs(subs_list) for f in weak_forms]
    return red_forms


def _solve_integrals(weak_forms):

    solved_map = dict()
    integrals = _find_integrals(weak_forms)
    for integral in integrals:
        if integral in solved_map:
            print("Skipping eval")
            continue
        if time not in integral.atoms(sp.Symbol):
            res = integral.eval_numerically()
            solved_map[integral] = res
        else:
            d_int = sp.Dummy()

    subs_forms = [f.subs(solved_map) for f in weak_forms]
    return subs_forms


def _reduce_kernel(integral):
    """ Move all independent terms outside the integral """
    kernel, (sym, a, b) = integral.args
    kernel = kernel.expand()
    if isinstance(kernel, sp.Add):
        new_int = sp.Add(*[_reduce_kernel(FakeIntegral(addend, (sym, a, b)))
                           for addend in kernel.args])

    elif isinstance(kernel, sp.Mul):
        dep_args = []
        indep_args = []
        for arg in kernel.args:
            if sym not in arg.atoms(sp.Symbol):
                indep_args.append(arg)
            else:
                dep_args.append(arg)

        new_int = sp.Mul(*indep_args) * FakeIntegral(sp.Mul(*dep_args),
                                                     (sym, a, b))

    return new_int