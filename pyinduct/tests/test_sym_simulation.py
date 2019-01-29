import pyinduct as pi
import pyinduct.sym_simulation as ss
import unittest
import sympy as sp


class DerivativeHandling(unittest.TestCase):

    def setUp(self):
        w = ss.get_weight()
        u = ss.get_input(ss.time, ss.space[0])
        f = ss.get_field_variable(ss.space[0])
        g = ss.get_test_function(ss.space[0], ss.space[1])

        self.w_dt = w.diff(ss.time)
        self.w_ddt = w.diff(ss.time, 2)
        self.w_d3t = w.diff(ss.time, 3)

        self.g_dz01 = g.diff(ss.space[0], 1, ss.space[1], 2)
        self.f_dz0 = f.diff(ss.space[0], 2)
        self.u_dtz0 = u.diff(ss.space[0], 2, ss.time, 2)

        self.expressions = sp.Matrix([
            self.w_dt * self.w_ddt + self.w_d3t,
            self.w_dt ** 2 + 5 * self.g_dz01 + sp.cos(self.f_dz0),
            self.w_dt * self.g_dz01 + self.u_dtz0,
        ])

    def test_find_derivative(self):
        # scalar targets
        res = ss._find_derivatives(self.expressions, ss.time)
        self.assertTrue(res.issubset({self.w_dt, self.w_ddt, self.w_d3t,
                                      self.u_dtz0}))

        res = ss._find_derivatives(self.expressions, ss.space[0])
        self.assertTrue(res.issubset({self.g_dz01, self.f_dz0, self.u_dtz0}))

        res = ss._find_derivatives(self.expressions, ss.space[1])
        self.assertTrue(res.issubset({self.g_dz01}))

        res = ss._find_derivatives(self.expressions, ss.space[2])
        self.assertFalse(res)

        # multiple targets
        res = ss._find_derivatives(self.expressions, (ss.time, ss.space[0]))
        self.assertTrue(
            res.issubset({self.w_dt, self.w_ddt, self.w_d3t, self.g_dz01,
                          self.f_dz0, self.u_dtz0}))

    def test_convert_higher_derivative(self):
        # Error cases
        with self.assertRaises(ValueError):
            # symbol not present in derivative
            ss._convert_higher_derivative(self.expressions, self.w_dt,
                                          sym=ss.space[0])

        with self.assertRaises(ValueError):
            # mixed derivative without symbol
            ss._convert_higher_derivative(self.expressions, self.g_dz01)

        res = ss._convert_higher_derivative(self.expressions, self.w_d3t)
        print(res)
