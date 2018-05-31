"""
New simulation module approach that makes use of sympy for expression handling
"""

import sympy as sp
import numpy as np

time = sp.symbols("t")
space = sp.symbols("z:3")


class FieldVariable(sp.Function):
    pass


class InnerProduct(sp.Expr):
    """ An unevaluated Inner Product on L2

    Args:
        a: Left Operand .
        b: Right Operand .

    """
    is_complex = True

    def __new__(cls, left, right):
        # identify free variables of arguments
        vars = left.atoms(sp.Symbol).union(right.atoms(sp.Symbol))

        # extract spatial coordinates
        vars = vars.intersection(space)

        # construct integrals over remaining coordinates
        exp = sp.conjugate(left) * right
        for dim in vars:
            exp = sp.Integral(exp, dim)
        return exp

    @property
    def left(self):
        return self.args[0]

    @property
    def right(self):
        return self.args[1]

    def doit(self, **hints):
        pass
