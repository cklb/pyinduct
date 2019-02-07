import pyinduct as pi
import pyinduct.sym_simulation as ss
import unittest
import sympy as sp


class TestSymStateSpace(unittest.TestCase):

    def setUp(self):
        self.t = ss.time
        self.x = sp.Matrix(ss.get_weights(10))
        self.u = sp.Matrix([ss.get_input(self.t) for n in range(5)])
        self.rhs = (sp.Matrix([sp.cos(x) for x in self.x])
                    + sp.ones(len(self.x), len(self.u)) @ self.u
        )

    def test_init(self):
        sys = ss.SymStateSpace(None, None, None, None, None, None)

    def test_sympy_fuckup(self):
        t = sp.Symbol("t")
        f = sp.Function("f", real=True)(t)

        f_str = sp.srepr(f)
        g = sp.sympify(f_str)

        assert hash(f) == hash(g)
        assert f == g

    def test_dump(self):
        dummies = ss._dummify_system(self.rhs, self.x, self.u)
        sys_1 = ss.SymStateSpace(*dummies, self.x, self.u, tuple())
        f = "test.sys"
        sys_1.dump(f)
        sys_2 = ss.SymStateSpace.from_file(f)
        self._check_compatibility(sys_1, sys_2)

    def _check_compatibility(self, sys1, sys2):
        self.assertTrue(id(sys1) != id(sys2))
        for x1, x2 in zip(sys1.orig_state, sys2.orig_state):
            # self.assertTrue(id(x1) != id(x2))
            self.assertTrue(hash(x1) == hash(x2))
            self.assertTrue(x1 == x2)

    def test_from_file(self):
        f = "test.sys"
        # f = "../examples/test_sys"
        sys1 = ss.SymStateSpace.from_file(f)
        sys2 = ss.SymStateSpace.from_file(f)
        self._check_compatibility(sys1, sys2)
        # sys1 = ss.SymStateSpace.from_file(f)
        # sys2 = ss.SymStateSpace.from_file(f)
        # self._check_compatibility(sys1, sys2)


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
