"""
New simulation module approach that makes use of sympy for expression handling
"""

from copy import copy
import sympy as sp
import symengine as se
from time import clock
from sympy.utilities.lambdify import implemented_function
from sympy.functions.special.polynomials import jacobi
import numpy as np
from tqdm import tqdm

from .registry import get_base
from .core import Domain, Function

time = se.symbols("t", real=True)
space = se.symbols("z:3", real=True)


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
    w = se.Function("{}_{}".format(_weight_letter, _weight_cnt),
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
    f = se.Function("{}_{}".format(_function_letter, _function_cnt),
                    real=True)(sym)
    _function_cnt += 1
    return f


_test_function_cnt = 0
_test_function_letter = "_g"


def get_test_function(*symbols):
    global _test_function_cnt
    g = se.Function("{}_{}".format(_test_function_letter, _test_function_cnt),
                    real=True)(*symbols)
    _test_function_cnt += 1
    return g


_input_cnt = 0
_input_letter = "_u"


def get_input():
    global _input_cnt
    u = se.Function("{}_{}".format(_input_letter, _input_cnt),
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
        self._cb = se.lambdify(weights,
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


class InnerProduct(se.Expr):
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
        # exp = sp.conjugate(left) * right
        exp = left * right
        for dim in variables:
            exp = sp.Integral(exp, (dim, bounds[0], bounds[1]))
        return exp

    @property
    def left(self):
        return self.args[0]

    @property
    def right(self):
        return self.args[1]

    def doit(self, **hints):
        pass


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

    boundary_positions = []
    ess_bcs = []
    nat_bcs = []
    ess_pairs = []
    ess_idxs = []

    base = get_base(base_lbl)

    # identify essential and natural boundaries
    for cond in boundary_conditions:
        lhs = cond.args[0]
        expr = lhs.args[0]
        assert sym in lhs.args[1]
        assert len(lhs.args[2]) == 1
        # extract the position where the dirichlet condition is given
        pos = next(iter(lhs.args[2]))
        boundary_positions.append(pos)
        pair = (cond, pos)

        if isinstance(expr, sp.Derivative):
            # only dirichlet boundaries require extra work
            nat_bcs.append(pair)
        else:
            ess_bcs.append(pair)

    # find the corresponding functions for the essential boundaries
    for cond, pos in ess_bcs:
        # identify all base fractions that differ from zero
        for idx, func in enumerate(base):
            if isinstance(func, Function):
                res = func(pos)
            elif isinstance(func, se.Basic):
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
        x_ess += cond.rhs * d_func
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
        variables = se.symbols("_m _n")
        _f = variables[0]*sym + variables[1]
    elif mode == "trig":
        variables = se.symbols("_a _b")
        _f = se.cos(variables[0]*sym + variables[1])
    # elif mode == "exp":
    #     variables = sp.symbols("_c _b")
    #     _f = sp.cos(variables[0]*sym + variables[1])
    else:
        raise NotImplementedError

    eqs = [_f.subs(pos_a) - 1, _f.subs(pos_b - 0)]
    sol = se.solve(eqs, variables, dict=True)[0]
    res = _f.subs(sol)

    return res


def substitute_approximations(weak_form, mapping):

    ext_form, ext_mapping = expand_kth_terms(weak_form, mapping)

    def wrap_expr(expr, *sym):
        def wrapped_func(*_sym):
            return expr.subs(list(zip(sym, _sym)))

        return wrapped_func

    def gen_func_subs_pair(func, expr):
        if isinstance(func, se.Function):
            a = func.func
            args = func.args
        elif isinstance(func, se.Subs):
            a = func
            if isinstance(func.args[0], se.Derivative):
                args = func.args[0].args[0].args
            elif isinstance(func.args[0], se.Function):
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
    for eq in tqdm(ext_form):
        rep_eq = eq
        for pair in tqdm(rep_list):
            # print("Substituting pair:")
            # print(pair)
            rep_eq = rep_eq.replace(*pair)
            # if u1_t in rep_eq.atoms(sp.Function):
            # print(rep_eq.atoms(sp.Derivative))
            # print("Result:")
            # print(rep_eq)

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
