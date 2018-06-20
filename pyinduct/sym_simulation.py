"""
New simulation module approach that makes use of sympy for expression handling
"""

from time import clock
from copy import copy
import warnings
import sympy as sp
# import symengine as sp
from sympy.utilities.autowrap import ufuncify
import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm

from matplotlib import pyplot as plt

from .registry import get_base
from .core import (Domain, Function, Base, domain_intersection, integrate_function,
                   project_on_base, EvalData)

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


def get_parameter(param):
    return _parameters.get(param, None)


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


def build_lag1st(sym, start, mid, end):
    if start == mid:
        obj = sp.Piecewise((0, sym < mid),
                           (1 - (sym - mid)/(end - mid), sym < end),
                           (0, sym >= end)
                           )
    elif mid == end:
        obj = sp.Piecewise((0, sym < start),
                           ((sym - start)/(mid - start), sym <= mid),
                           (0, sym > mid)
                           )
    else:
        obj = sp.Piecewise((0, sym < start),
                           ((sym - start)/(mid - start), sym < mid),
                           (1, sym == mid),
                           (1 - (sym - mid)/(end - mid), sym < end),
                           (0, sym >= end)
                           )
    return obj


def create_lag1st_base(sym, bounds, num):
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


class LumpedApproximation:

    def __init__(self, symbols, ess_expr, nat_expr, weights, base_map, bcs):
        self._syms = symbols
        self._ess_expr = ess_expr
        self._nat_expr = nat_expr
        self._weights = weights
        self._base_map = base_map
        self._bcs = bcs

        # substitute all known functions and symbols and generate callback
        self._cb = sp.lambdify(self._syms + self.weights,
                               self.expression.subs(get_parameters()),
                               modules="numpy")

    @property
    def expression(self):
        return self._ess_expr + self._nat_expr

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
            return self._cb(*args)
        if kwargs:
            return self._cb(*[kwargs[w] for w in self._weights])

    def approximate_function(self, func):
        """
        Project the given function into the subspace of this Approximation

        Args:
            func(sp.Function): Function expression to approximate
        """
        if isinstance(func, sp.Basic):
            hom_func = (func - self._ess_expr).subs(get_parameters())
            cb = sp.lambdify(hom_func.free_symbols, hom_func)
            # else:
            #     retval = hom_func.evalf()
            #     def _dummy(*args):
            #         return retval
            #     cb = _dummy
        else:
            # lets hope that it is callable somehow
            cb = func

        weights = project_on_base(Function(cb), Base(self.base))
        return weights

    def get_spatial_approx(self, weights):
        expr = self.expression.subs(list(zip(self.weights, weights))).subs(get_parameters())
        return sp.lambdify(self._syms, expr, modules="numpy")


class InnerProduct(sp.Expr):
    """
    An Inner Product on L2

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
    Placeholder class that looks like an Integral

    The difference is that this class but won't try to integrate anything
    if not directly told to do so. Using this approach, integral expression
    can be handled and simplified more easily, until eventually they are
    solved numerically.
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
            if 0:
                reg = next(iter(interval))
                x_vals = np.linspace(float(a), float(b))
                plt.plot(x_vals, f(x_vals).T)
                plt.axhline(xmin=reg[0], xmax=reg[1], c="r")
                plt.show()

            res, err = integrate_function(f, interval)
        else:
            # kernel is zero on the whole domain
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


def create_approximation(syms, base_lbl, boundary_conditions, weights=None):
    """
    Create a lumped approximation of a distributed variable

    The approximation is performed as a product-ansatz using the functions
    registered under *base_lbl*. If essential boundaries are detected, they
    are homogenized, leading to a possible reduction of free weights in the
    resulting approximation.

    Args:
        syms((list of) Symbol(s)): Symbol(s) to use for approximation.
            (Dimension to discretise)
        base_lbl(str): Label of the base to use
        boundary_conditions(list): List of boundary conditions that have to be
            met by the approximation.

    Keyword Args:
        weights(list): If provided, weight symbols to use for the approximation.
            By default (None), a new set of weights will be created.

    Returns:
        LumpedApproximation: Object holding all information about the
        approximation.
    """
    if isinstance(syms, sp.Basic):
        syms = [syms]

    if len(syms) > 1:
        raise NotImplementedError("Higher dimensional cases are still missing")
    sym = syms[0]
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
        if isinstance(func, Function):
            d_func = get_function(sym)
            d_func.func._imp_ = staticmethod(func)
        else:
            d_func = func
        x_ess += cond[1] * d_func
        base_mapping[d_func] = func, True

    # extract shape functions which are to be kept
    nat_funcs = [func for idx, func in enumerate(base) if idx not in ess_idxs]

    # homogeneous part
    x_nat = 0
    if weights is None:
        weights = get_weights(len(nat_funcs))
    for idx, func in enumerate(nat_funcs):
        if isinstance(func, Function):
            d_func = get_function(sym)
            d_func.func._imp_ = staticmethod(func)
        else:
            d_func = func
        x_nat += weights[idx] * d_func
        base_mapping[d_func] = func, False

    x_approx = LumpedApproximation([sym],
                                   x_ess, x_nat,
                                   weights,
                                   base_mapping,
                                   boundary_conditions)

    return x_approx


def _classify_boundaries(sym, boundary_conditions):
    """
    Identify essential and natural boundaries

    Since Dirichlet boundaries have to be handled different, this function takes
    a list of Equation objects and classifies their type into essential
    (dirichlet) and natural (neumann or robin) boundary conditions.

    Note:
        For the detection to work, substitutions must not have been performed.
        Therefore, create expression like `Subs(x(z,t), z, 0)` in favour of
        `x(z,t).subs(z, 0)`.
        However, using symengine this will also fail.

    Args:
        sym(Symbol): Symbol, specifying the coordinate to classify for.
        boundary_conditions(list): List of equations, defining the boundaries.

    Returns:
        3-tuple of essential boundaries, natural boundaries and all boundary
        positions.
    """
    # TODO: Pass placeholder expr and directly search for that
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
    """
    Substitute placeholder functions with their approximations

    Firstly, all testfunctions are substituted, by adding a new equation for
    every provided entity. Afterwards, the approximations for all other symbols
    are performed.

    Args:
        weak_form (list): List of sympy expression, representing the weak
            formulation of a system.
        mapping (dict): Dict holding the substitution mappings

    Returns:
        Extended equation system with substituted mapping.
    """

    ext_form, ext_mapping = _expand_kth_terms(weak_form, mapping)

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
            rep_list += [gen_func_subs_pair(*bc.args) for bc in approx.bcs]
            rep_list.append(gen_func_subs_pair(sym, approx.expression))
        else:
            rep_list.append(gen_func_subs_pair(sym, approx))

    print(">>> substituting symbolic approximations")
    rep_eqs = []
    for eq in tqdm(ext_form):
    # for eq in ext_form:
        rep_eq = eq
        for pair in rep_list:
            rep_eq = rep_eq.replace(*pair, simultaneous=True)

        rep_eqs.append(rep_eq.doit())

    return rep_eqs


def _expand_kth_terms(weak_form, mapping):
    """
    Search for kth terms and expand the equation system by their mappings

    For now, a *k-th term* is recognized by its placeholder function being
    a test function. Since in the weak form, it acts as a placeholder for all
    test functions, an equation is added to the equation system for every entry
    in the value, belonging to a test-function key.

    Therefore, having one equation with two *k-th terms* and a mapping assigning
    10 function to each of them, yo will end up with 100 equations.
    """
    new_exprs = []
    new_mapping = {}
    kth_placeholders = {}

    # search for kth placeholders
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
            if isinstance(expr, Function):
                _g.func._imp_ = staticmethod(expr)
            else:
                mappings[_g] = expr

            r_tpl = symbol.func, _g.func
            new_eqs.append(equation.replace(*r_tpl))

    return new_eqs, mappings


def create_first_order_system(weak_forms):

    print(">>> identifying inputs")
    inputs = _find_inputs(weak_forms)
    sorted_inputs = sorted(inputs, key=lambda x: str(x))
    # sp.pprint(sorted_inputs)

    print(">>> identifying state components")
    state_elems = _find_weights(weak_forms)
    sorted_state = sp.Matrix(sorted(state_elems, key=lambda x: str(x)))
    # sp.pprint(sorted_state)

    print(">>> simplifying integrals")
    new_forms = _simplify_integrals(weak_forms)
    # sp.pprint(new_forms, num_columns=200)

    print(">>> substituting derivatives")
    new_forms, targets = _convert_derivatives(new_forms)
    # sp.pprint(new_forms, num_columns=200)

    print(">>> substituting parameters")
    new_forms = [form.subs(get_parameters()) for form in new_forms]
    # sp.pprint(new_forms, num_columns=200)

    new_forms = _solve_integrals(new_forms)
    # sp.pprint(new_forms, num_columns=200)

    print(">>> running remaining evaluations")
    new_forms = [form.doit() for form in new_forms]
    # sp.pprint(ss_form)

    print(">>> solving for targets")
    sol_dict = sp.solve(new_forms, targets)
    # sp.pprint(ss_form)

    print(">>> building statespace form")
    ss_form = sp.Matrix([sol_dict[s.diff(time)] for s in sorted_state])
    # sp.pprint(ss_form)

    return ss_form, sorted_state, sorted_inputs


def _convert_derivatives(new_forms):
    all_symbols = set()
    for f in new_forms:
        all_symbols.update(f.atoms(sp.Symbol))
    derivatives = _find_derivatives(new_forms, all_symbols)

    # derivatives = _find_derivatives(new_forms, time)
    # derivatives.update(_find_derivatives(new_forms, space))
    subs_list = []
    targets = set()
    for der in derivatives:
        new_forms, subs_pairs, target_set = _convert_higher_derivative(
            new_forms, der)
        subs_list += subs_pairs
        targets.update(target_set)
    return new_forms, targets


def _find_weights(new_forms):
    state_elems = set()
    for form in new_forms:
        funcs = form.atoms(sp.Function)
        for f in funcs:
            if _weight_letter in str(f) and f.args[0] == time:
                state_elems.add(f)
    return state_elems


def _find_inputs(weak_forms):
    """ Find system inputs """
    inputs = set()
    for eq in weak_forms:
        _inputs = [_inp for _inp in eq.atoms(sp.Function)
                   if _input_letter in str(_inp)]
        inputs.update(_inputs)

    return inputs


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
        sym = {sym}

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
            new_forms = weak_forms
            target_set.add(derivative)

    elif _input_letter in str(expr):
        raise NotImplementedError

    elif _function_letter in str(expr) or _test_function_letter in str(expr):
        # derive the associated callbacks
        if _function_letter in str(expr):
            d_func = get_function(*expr.args)

        elif _test_function_letter in str(expr):
            d_func = get_test_function(*expr.args)

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
        if integral != red_int:
            subs_list.append((integral, red_int))

    red_forms = [f.subs(subs_list) for f in weak_forms]
    return red_forms


def _solve_integrals(weak_forms):

    solved_map = dict()
    integrals = _find_integrals(weak_forms)
    print(">>> solving integrals")
    # for integral in integrals:
    for integral in tqdm(integrals):
        if integral in solved_map:
            print("Skipping eval")
            continue
        if time in integral.free_symbols:
            res = sp.integrate(integral.args[0], integral.args[1])
            if res.free_symbols.intersection(tuple(space)):
                raise ValueError
            solved_map[integral] = res
        else:
            res = integral.eval_numerically()
            solved_map[integral] = res

    tqdm.write(">>> substituting the solutions")
    subs_forms = [f.subs(solved_map) for f in tqdm(weak_forms)]
    return subs_forms


def _reduce_kernel(integral):
    """
    Move all independent terms outside the integral's kernel

    To reach this goal, firstly the kernel is expanded and split. Afterwards,
    independent term are moved aside. If the kernel still contains unknown
    expressions (such as weights and their derivatives) they are expanded into
    a taylor series and separated afterwards.
    The last step is only performed if `enable_approx` is `True` .

    Args:
        integral(FakeIntegral): Integral whose kernel is to be simplified.

    Returns:
        Sum of simplified integrals.

    """
    approx = get_parameter("enable_approx")

    kernel, (sym, a, b) = integral.args
    if time not in kernel.free_symbols:
        return integral

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

        if not indep_args:
            bad_args = {arg for arg in dep_args if time in arg.free_symbols}
            if not bad_args:
                return FakeIntegral(kernel, (sym, a, b))
            else:
                if approx:
                    arg_map = {arg: _approximate_term(arg, sym)
                               for arg in bad_args}
                else:
                    return FakeIntegral(kernel, (sym, a, b))

            dep_args = [arg.subs(arg_map) for arg in dep_args]

        new_kernel = sp.Mul(*dep_args)
        new_part = _reduce_kernel(FakeIntegral(new_kernel, (sym, a, b)))
        new_int = sp.Mul(*indep_args) * new_part
    else:
        print("Unable to handle '{}'".format(kernel))
        return FakeIntegral(kernel, (sym, a, b))

    return new_int


def _approximate_term(term, sym):
    """
    Series expansion of unseparable terms

    Args:
        term: Term to expand.
        sym: Coordinate to expand.

    Returns:
        Taylor expansion of the highest order supported by the involved
        function expressions.
    """
    n = get_parameter("approx_order")
    pos = get_parameter("approx_pos")

    warnings.warn("Approximating term {} at {}={} with order n={}".format(
        term, sym, pos, n))
    ser = sp.series(term, x=sym, x0=pos, n=n).removeO()
    return ser


def simulate_state_space(time_dom, rhs_expr, ics, input_dict, inputs, state):
    """
    Simulate an ODE system given its right-hand side

    """
    # build expressions
    input_mapping = {inp: sp.Dummy() for inp in inputs}
    state_mapping = {s: sp.Dummy() for s in state}
    subs_map = {**input_mapping, **state_mapping}
    dummy_rhs = sp.Matrix([eq.subs(subs_map) for eq in rhs_expr])

    args = [[input_mapping[inp] for inp in inputs],
            [state_mapping[st] for st in state]]
    rhs_cb = sp.lambdify(args, expr=dummy_rhs, modules="numpy")

    rhs_jac = dummy_rhs.jacobian(args[1])
    jac_cb = sp.lambdify(args, expr=rhs_jac, modules="numpy")

    def _rhs(t, y):
        u = [input_dict[inp](t, y) for inp in inputs]
        y_dt = np.ravel(rhs_cb(u, y))
        return y_dt

    def _jac(t, y):
        u = [input_dict[inp](t, y) for inp in inputs]
        jac = jac_cb(u, y)
        return jac

    # build initial state
    init_weights = dict()
    for key, val in ics.items():
        _weight_set = key.approximate_function(val)
        new_d = {(lbl, w) for lbl, w in zip(key.weights, _weight_set)}
        init_weights.update(new_d)

    y0 = [init_weights[lbl] for lbl in state]

    # simulate
    res = solve_ivp(fun=_rhs,
                    y0=y0,
                    t_span=time_dom.bounds,
                    t_eval=time_dom.points,
                    jac=_jac,
                    # method="BDF"
                    )
    if not res.success:
        warnings.warn("Integration failed at t={} with '{}'".format(
            res.t[-1], res.message))

    return res


def _sort_weights(weights, state, approximations):
    """ Coordinate a given weight set with approximations """
    weight_dict = dict()
    state_elements = list(state)
    for approx in approximations:
        a_lbls = approx.weights
        a_weights = []
        for lbl in a_lbls:
            idx = state_elements.index(lbl)
            a_weights.append(weights[idx])

        weight_dict[approx] = np.array(a_weights)

    return weight_dict


def _evaluate_approximations(weight_dict, approximations, temp_dom, spat_doms):
    """
    Evaluate approximations on the given grids


    """
    if isinstance(spat_doms, Domain):
        spat_doms = [spat_doms]

    results = []
    all_coords = [dom.points for dom in spat_doms]
    all_coords.append(np.array(range(len(temp_dom))))
    all_dims = [len(dom) for dom in all_coords]
    grids = np.meshgrid(*all_coords, indexing="ij")
    r_grids = [grid.ravel() for grid in grids]

    for approx in approximations:
        weight_mat = weight_dict[approx]
        temp_dim = weight_mat.shape[0]
        args = np.zeros(len(grids) - 1 + temp_dim)
        res = np.zeros(len(r_grids[0]))
        for coord_idx in range(len(r_grids[0])):
            # fill spatial parameters
            args[:-temp_dim] = [r_grid[coord_idx] for r_grid in r_grids[:-1]]
            args[-temp_dim:] = weight_mat[:, r_grids[-1][coord_idx]]
            res[coord_idx] = approx(*args)

        # per convention the time axis comes first
        out_data = np.moveaxis(res.reshape(all_dims), -1, 0)
        data = EvalData(input_data=[temp_dom] + spat_doms,
                        output_data=out_data)
        results.append(data)

    return results
