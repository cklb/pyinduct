import numpy as np
from copy import copy

import sympy as sp
import pyinduct as pi
import pyqtgraph as pg


def calc_exp_base(parameter, order, spat_domain, debug=False):
    """
    Calculate the exponential base for the string with mass.

    Args:
        parameter: physical parameters
        order: grade of differential smoothness
        spat_domain: spatial domain of the problem
        debug: Show plot of eigenvalues and generated functions

    Returns:
        pi.Base: Exponential Base
    """
    # create symbolic solution
    tau, sigma, m, t, z = sp.symbols("tau sigma m t z")
    s = sp.symbols("s", complex=True)
    c1, c2 = sp.symbols("c_1, c_2")
    p1 = tau * s
    p2 = -tau * s
    w0 = c1 * sp.exp(p1 * z) + c2 * sp.exp(p2 * z)

    # w0(0) = 1
    bc0 = 1 - w0.subs(z, 0)
    w0 = w0.subs(c1, sp.solve(bc0, c1)[0])

    # bc at z=0
    bc1 = sigma * w0.diff(z).subs(z, 0) - m * s ** 2 * w0.subs(z, 0)
    w0 = w0.subs(c2, sp.solve(bc1, c2)[0])

    # substitute constants
    w0 = w0.subs([(m, parameter.m),
                  (tau, parameter.tau),
                  (sigma, parameter.sigma)])

    # equations
    char_eq = (parameter.sigma * w0.diff(z).subs(z, 1))
    char_eq_num = sp.lambdify(s, char_eq, modules="numpy")

    # find eigenvalues
    grid = [np.linspace(-1, 1), np.linspace(0, 100)]
    roots = pi.find_roots(char_eq_num,
                          grid,
                          cmplx=True)

    # automatically check for double roots, but comes with numeric problems
    # char_eq_ds = char_eq.diff(s)
    # char_eq_ds_num = sp.lambdify(s, char_eq_ds, modules="numpy")
    # roots_ds = pi.find_roots(char_eq_ds_num,
    #                          grid,
    #                          sort_mode="component",
    #                          cmplx=True)
    # pos_idx_ds = np.imag(roots) >= 0
    # eig_values_ds = roots_ds[pos_idx_ds]
    # pi.visualize_roots(eig_values_ds, grid, char_eq_ds_num, cmplx=True)

    # clear numeric jitter
    roots = 1j*np.imag(roots)
    eig_values = roots[:order]

    if debug:
        grid = [np.linspace(-1, 1), np.linspace(0, 100)]
        pi.visualize_roots(eig_values, grid, char_eq_num, cmplx=True)

    def w0_k_factory(eigenvalue, multiplicity=1, derivative_order=0):
        """
        Factory that produces derivatives of w_0(z)

        Args:
            eigenvalue: Eigenvalue.
            multiplicity: Multiplicity of the given eigenvalue *s* .
            derivative_order: Derivative of w0(sk, z) to return.

        Returns:

        """
        w0_derivative = w0.diff(s, multiplicity - 1).diff(z, derivative_order)
        # w0_s = sp.re(w0_derivative.subs(s, eigenvalue))
        w0_s = w0_derivative.subs(s, eigenvalue)
        return sp.lambdify(z, w0_s, modules="numpy")

    fracs = []
    for sk in eig_values:
        multiplicity = 1

        # TODO remove hardcoded check and better look into roots of char_ds
        # if sk in eig_values_ds:
        # if np.isclose(sk, 0):
            # this will create the zero function -> useless
            # multiplicity += 1

        for m in range(1, multiplicity + 1):
            frac = pi.Function(domain=spat_domain.bounds,
                               nonzero=spat_domain.bounds,
                               eval_handle=w0_k_factory(sk, multiplicity=m),
                               derivative_handles=[
                                   w0_k_factory(sk,
                                                multiplicity=m,
                                                derivative_order=1),
                                   w0_k_factory(sk,
                                                multiplicity=m,
                                                derivative_order=2)])
            fracs.append(frac)

    if debug:
        pi.visualize_functions(fracs)

    return pi.Base(fracs), eig_values


def calc_flat_base(eig_values, pseudo_domain, debug=False):
    """
    Generates the basis for the restriction of the flat output on a certain
    interval.

    Args:
        eig_values: Eigenvalues to use.
        pseudo_domain: Spatial domain to use.

    Returns:

    """
    s, z = sp.symbols("s z")
    exp_func = sp.exp(s * z)

    def exp_func_factory(eigenvalue, multiplicity=1, derivative_order=0):
        """
        Factory for exponential functions.

        Args:
            eigenvalue: Eigenvalue
            multiplicity: Geometric multiplicity of the eigenvalue.
            derivative_order: Differential order

        Returns:

        """
        exp_derivative = exp_func.diff(s, multiplicity - 1,
                                       z, derivative_order)
        exp_s = exp_derivative.subs(s, eigenvalue)
        return sp.lambdify(z, exp_s, modules="numpy")

    fracs = []
    for sk in eig_values:
        multiplicity = 1

        # TODO remove hardcoded check and better look into roots of char_ds
        # if sk in eig_values_ds:
        # if np.isclose(sk, 0):
        #     multiplicity += 1

        for m in range(1, multiplicity + 1):
            frac = pi.Function(domain=pseudo_domain.bounds,
                               nonzero=pseudo_domain.bounds,
                               eval_handle=exp_func_factory(sk, multiplicity=m),
                               derivative_handles=[
                                   exp_func_factory(sk,
                                                    multiplicity=m,
                                                    derivative_order=1),
                                   exp_func_factory(sk,
                                                    multiplicity=m,
                                                    derivative_order=2)])
            fracs.append(frac)

    if debug:
        pi.visualize_functions(fracs, points=1e3)

    return pi.Base(fracs)


class SWMStateVector(pi.ComposedFunctionVector):
    """
    vector that represents the state of the string with mass,
    containing (x'(z, t), x.(z, t), x(0, t), x.(0, t)) at t=0
    """

    def __init__(self, functions, scalars, origin_label):
        if len(functions) != 2 or len(scalars) != 2:
            raise TypeError("wrong dimensions!")

        super().__init__(functions, scalars)
        self.orig_lbl = origin_label

    @property
    def funcs(self):
        return self.members["funcs"]

    @property
    def scalars(self):
        return self.members["scalars"]

    def transformation_hint(self, info, target):
        """
        returns ready to use handle for weight transformation from pure Function
        Base, otherwise super method is returned.
        """
        if target is False:
            raise NotImplementedError

        if target and isinstance(info.src_base[0], pi.Function):
            if info.dst_order > info.src_order - 1:
                return None, None

            if info.src_lbl == self.orig_lbl:
                return self._transform_from_functions_factory(info), None
            else:
                # create transform from someone who knows how to convert
                name = info.src_lbl + "_state"

                # from current src to assistant system
                assistant_info = copy(info)
                assistant_info.dst_lbl = name
                assistant_info.dst_base = pi.get_base(name)
                assistant_info.dst_order = info.dst_order

                # from assistant system to dst
                target_info = copy(info)
                target_info.src_lbl = name
                target_info.src_base = pi.get_base(name)
                target_info.src_order = info.dst_order

                super_handle, super_hint = super().transformation_hint(
                    target_info, True)
                return super_handle, assistant_info

        # fallback method of super class
        return super().transformation_hint(info, target)

    @staticmethod
    def _transform_from_functions_factory(info):
        """
        calculates transformation that converts weights from src to dst basis
        """
        src_dim = info.src_base.fractions.size
        dst_dim = info.dst_base.fractions.size

        if dst_dim % 2 != 0:
            return None

        mat = np.zeros((dst_dim * (info.dst_order + 1),
                        src_dim * (info.src_order + 1)))

        # generate core of mapping
        core = np.zeros((dst_dim, 2 * src_dim))
        for i in range(src_dim):
            core[2 * i, i] = 1
            core[2 * i - 1, i + src_dim] = 1

        # fill with copies until needed order is accomplished
        for o in range(info.dst_order + 1):
            mat[o * dst_dim:(1 + o) * dst_dim,
            o * 2 * src_dim:(1 + o) * 2 * src_dim] = core

        def handle(weights):
            return np.dot(mat, weights)

        return handle

    # def scale(self, factor):
    #     return HCStateVector(np.array([func.scale(factor) for func in self.members["funcs"]]),
    #                          np.array([scal * factor for scal in self.members["scalars"]]),
    #                          self.orig_lbl)

    def evaluation_hint(self, values):
        return self(values)


def create_function_vectors(label, eig_vals=None, zero_padding=False):
    """
    builds a complete set (basis) of function vectors, length will be 2 times
    the length of the input.

    Args:
    :param label: label of function set that forms the basis of the state
        approximation
    :param eig_vals: eigenvalues, corresponding to the shapefunctions,
        described by label
    :param zero_padding: perform padding with zeros to construct function vector
        from simulation shapefunctions
    :return:
    """
    entries = []
    funcs = pi.get_base(label).fractions
    funcs_dz = pi.get_base(label).derive(1).fractions
    zero_func = pi.Function(lambda z: 0, nonzero=(0, 0))

    for idx, (func, func_dz) in enumerate(zip(funcs, funcs_dz)):
        if zero_padding:
            entries.append(SWMStateVector(functions=(func_dz, zero_func),
                                          scalars=(func(0), 0),
                                          origin_label=label))
            entries.append(SWMStateVector(functions=(zero_func, func),
                                          scalars=(0, func(0)),
                                          origin_label=label))
        else:
            entries.append(SWMStateVector(functions=(func_dz,
                                                     func.scale(eig_vals[idx])),
                                          scalars=(func(0),
                                                   eig_vals[idx] * func(0)),
                                          origin_label=label))

            # construct extra function vector for double eigenvalue at s=0
            if np.isclose(eig_vals[idx], 0, atol=1e-3) and False:
                # TODO verify what is correct in the end (probably both will deliver the same result)
                if 1:
                    # derive only 2nd part of state vector
                    entries.append(HCStateVector(functions=(zero_func, func),
                                                 scalars=(0, func(0)),
                                                 origin_label=label))
                else:
                    # derive whole state vector
                    # TODO calculate lim(d/ds w=(s, z)) for s->0
                    entries.append(HCStateVector(functions=(extra_funcs[1], func),
                                                 scalars=(extra_funcs[0](0), func(0)),
                                                 origin_label=label))

    return pi.Base(entries)


def calc_controller(exp_base_lbl, flat_base_lbl, parameter):
    """
    Generate the control law for a certain dynamic.

    Args:
        exp_base_lbl:
        flat_base_lbl:

    Returns:

    """
    y = pi.FieldVariable(flat_base_lbl, weight_label=exp_base_lbl)
    y_dz = y.derive(spat_order=1)
    y_ddz = y.derive(spat_order=2)

    p = parameter
    terms = [
        pi.ScalarTerm(y_ddz(0), scale=0),
        # pi.ScalarTerm(y_ddz(-p.tau), scale=(1-p.alpha)),
        # pi.ScalarTerm(y_dz(p.tau), scale=(p.sigma * p.tau / 2 - p.kappa1)),
        # pi.ScalarTerm(y_dz(-p.tau),
        #               scale=(p.alpha * p.kappa1 - p.sigma * p.tau / 2)),
        # pi.ScalarTerm(y(p.tau), scale=(-p.kappa0)),
        # pi.ScalarTerm(y(-p.tau), scale=(p.alpha * p.kappa0)),
    ]

    law = pi.WeakFormulation(terms, name="flat_law")
    controller = pi.Controller(law)
    return controller

if __name__ == '__main__':
    # Tests
    params = pi.Parameters(sigma=1, tau=1, m=1,
                           alpha=1, kappa1=1, kappa0=1)
    spat_domain = pi.Domain(bounds=(0, 1), num=50)

    exp_base, eig_vals = calc_exp_base(params, 10, spat_domain, debug=False)
    pi.register_base("exp_base", exp_base)

    pseudo_domain = pi.Domain(bounds=(-params.tau, params.tau), num=50)
    flat_base = calc_flat_base(eig_vals, pseudo_domain, debug=True)
    pi.register_base("flat_base", flat_base)

    cont = calc_controller("exp_base", "flat_base", params)
