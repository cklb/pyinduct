import pyinduct as pi
import pyinduct.sym_simulation as ss
import unittest
import sympy as sp


class DerivativeHandling(unittest.TestCase):

    def setUp(self):
        self.symbols = sp.symbols("t z x y")
        self.expr = sp.Function("f")(*self.symbols)
        self.derivatives = []
        self.derivatives.append(self.expr.diff(self.symbols[0], 2))
        self.derivatives.append(self.derivatives[-1].diff(self.symbols[1], 2))

    def test_find_derivative(self):
        res = ss._find_derivatives(self.derivatives, self.symbols[0])
