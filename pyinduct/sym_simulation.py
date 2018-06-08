"""
New simulation module approach that makes use of sympy for expression handling
"""

from copy import copy
import sympy as sp
from sympy.functions.special.polynomials import jacobi
import numpy as np

from .core import Domain

time = sp.symbols("t", real=True)
space = sp.symbols("z:3", real=True)


_symbol_cnt = 0


def get_weights(n, letter="c"):
    cnt = getattr(get_weights, "_cnt", 0)
    if n < 0:
        raise ValueError
    c = [sp.symbols("{}{}".format(letter, cnt + i),
                    real=True,
                    cls=sp.Function)(time)
         for i in range(n)]
    setattr(get_weights, "_cnt", cnt + n)
    return c


def get_derivative(n, letter="d"):
    cnt = getattr(get_derivative, "_cnt", 0)
    if n < 0:
        raise ValueError
    d = [sp.symbols("{}{}".format(letter, cnt + i),
                    real=True,
                    cls=sp.Function)(time)
         for i in range(n)]
    setattr(get_derivative, "_cnt", cnt + n)
    return d


class FieldVariable(sp.Function):
    pass


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


def create_approximation(sym, funcs, bounds):
    """
    Create an approximation basis.
    """

    ess_bcs = []
    ess_idxs = []
    # identify essential boundaries and their corresponding functions
    for cond in bounds:
        expr = cond.lhs.args[0]
        if isinstance(expr, sp.Derivative):
            # only dirichlet boundaries require extra work
            continue

        assert sym in cond.lhs.args[1]
        assert len(cond.lhs.args[2]) == 1
        pos = next(iter(cond.lhs.args[2]))

        for idx, func in enumerate(funcs):
            if func.subs(sym, pos) != 0:
                ess_bcs.append((cond, func))
                ess_idxs.append(idx)

    assert len(ess_bcs) <= len(bounds)

    # extract shape functions that are to be excluded
    x_ess = sp.Add(*[cond.rhs * func for cond, func in ess_bcs])

    # extract shape functions are to be kept
    nat_funcs = [func for idx, func in enumerate(funcs) if idx not in ess_idxs]

    # define approximated system variable
    c = get_weights(len(nat_funcs))
    x_nat = sp.Add(*[c * f for c, f in zip(c, nat_funcs)])

    x_comp = x_ess + x_nat

    return x_comp, nat_funcs

