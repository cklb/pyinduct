import numpy as np

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
    mass = parameter.m
    tau = parameter.tau
    sigma = parameter.sigma
    c = 1/tau

    def char_eq(s):
        return (domain.bounds[1] * (mass*s**2/sigma
                                    - s/c)
                + s/c * np.exp(s/c*domain.bounds[1]))

    # find eigenvalues
    grid = [np.linspace(-1, 10), np.linspace(-100, 100)]
    eig_values = pi.find_roots(char_eq,
                               grid,
                               cmplx=True)

    if debug:
        pi.visualize_roots(eig_values, grid, char_eq, cmplx=True)

    def w0_k_factory(sk, derivative_order=0):
        """
        Factory that produces derivatives of w_0(z)

        Args:
            sk:
            derivative_order:

        Returns:

        """
        def w0(z):
            return z * (mass * sk**2/sigma - sk/c) + np.exp(sk/c * z)

        def w0_dz(z):
            return (mass * sk**2/sigma - sk/c) + sk/c * np.exp(sk/c * z)

        def w0_ddz(z):
            return (sk/c)**2 * np.exp(sk/c * z)

        if derivative_order == 0:
            return w0
        elif derivative_order == 1:
            return w0_dz
        elif derivative_order == 2:
            return w0_ddz
        else:
            raise ValueError

    fracs = []
    for sk in eig_values[:order]:
        frac = pi.Function(domain=spat_domain.bounds,
                           nonzero=spat_domain.bounds,
                           eval_handle=w0_k_factory(sk),
                           derivative_handles=[w0_k_factory(sk, 1),
                                               w0_k_factory(sk, 2)])
        fracs.append(frac)

    if debug:
        pi.visualize_functions(fracs)

    return pi.Base(fracs)

if __name__ == '__main__':
    # Tests
    params = pi.Parameters(sigma=1, tau=1, m=1)
    domain = pi.Domain(bounds=(0, 1), num=50)
    exp_base = calc_exp_base(params, 10, domain)
