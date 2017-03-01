import numpy as np

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
    grid = [np.linspace(-1, 1), np.linspace(-1, 100)]
    roots = pi.find_roots(char_eq_num,
                          grid,
                          sort_mode="component",
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

    pos_idx = np.imag(roots) >= 0
    eig_values = roots[pos_idx]

    if debug:
        grid = [np.linspace(-1, 1), np.linspace(-1, 100)]
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
        w0_s = w0_derivative.subs(s, eigenvalue)
        return sp.lambdify(z, w0_s, modules="numpy")

    fracs = []
    for sk in eig_values[:order]:
        multiplicity = 1

        # TODO remove hardcoded check and better look into roots of char_ds
        # if sk in eig_values_ds:
        if np.isclose(sk, 0):
            # this will create the zero function -> useless
            multiplicity += 1

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
        if np.isclose(sk, 0):
            multiplicity += 1

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
        pi.ScalarTerm(y_ddz(-p.tau), scale=(1-p.alpha)),
        pi.ScalarTerm(y_dz(p.tau), scale=(p.sigma * p.tau / 2 - p.kappa1)),
        pi.ScalarTerm(y_dz(-p.tau),
                      scale=(p.alpha * p.kappa1 - p.sigma * p.tau / 2)),
        pi.ScalarTerm(y(p.tau), scale=(-p.kappa0)),
        pi.ScalarTerm(y(-p.tau), scale=(p.alpha * p.kappa0)),
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
