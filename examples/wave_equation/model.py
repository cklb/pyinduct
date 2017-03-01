import pyinduct as pi
import pyinduct.hyperbolic as hyperbolic

import pyqtgraph as pg


def calc_weak_form(parameter, base_lbl, domain, input_object):
    """
    Constitute the weak formulation of the string with mass.

    Args:
        parameter (pi.Parameter): Physical parameters.
        base_lbl (str): Label of the approximation base to use.
        domain (pi.Domain): Domain of the problem.
        input_object (pi.Input): Input object for the equation

    Returns:
        pi.WeakForm: Weak formulation of the system
    """
    # weak form of the model equations
    x = pi.FieldVariable(base_lbl)
    x_ddt = x.derive(temp_order=2)
    x_dz = x.derive(spat_order=1)

    psi = pi.TestFunction(base_lbl)
    psi_dz = psi.derive(order=1)

    terms = [
        pi.IntegralTerm(pi.Product(x_ddt, psi),
                        limits=domain.bounds,
                        scale=parameter.sigma * parameter.tau**2),
        pi.ScalarTerm(pi.Product(x_ddt(0), psi(0)),
                      scale=parameter.m),
        pi.IntegralTerm(pi.Product(x_dz, psi_dz),
                        limits=domain.bounds,
                        scale=parameter.sigma),
        pi.ScalarTerm(pi.Product(pi.Input(input_object), psi(domain.bounds[1])),
                      scale=-parameter.sigma)
    ]

    return pi.WeakFormulation(terms, name="swm-{}-approx".format(base_lbl))


if __name__ == '__main__':
    # settings
    params = pi.Parameters(sigma=1, tau=1, m=1)
    spat_dom = pi.Domain(bounds=(0, 1), num=50)
    temp_dom = pi.Domain(bounds=(0, 10), num=50)

    # test base
    nodes, base = pi.cure_interval(pi.LagrangeFirstOrder, spat_dom.bounds, 10)
    pi.register_base("sim_base", base)

    # test input
    u = hyperbolic.feedforward.FlatString(y0=0, y1=1,
                                          z0=spat_dom.bounds[0],
                                          z1=spat_dom.bounds[1],
                                          t0=1, dt=3,
                                          params=params)

    # get weak form
    system = calc_weak_form(params, "sim_base", spat_dom, u)

    # initial conditions
    ic = [
        pi.Function(lambda z: 0),  # x(z, 0)
        pi.Function(lambda z: 0),  # dx_dt(z, 0)
    ]

    # simulate system
    data = pi.simulate_system(system, ic, temp_dom, spat_dom)

    an_plot = pi.PgAnimatedPlot(data)
    # sf_plot = pi.PgSurfacePlot(data[0])
    pg.QAPP.exec_()
