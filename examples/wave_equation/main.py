import pyinduct as pi
import pyqtgraph as pg

from feedforward import FlatFeedForward
from control import calc_exp_base, calc_flat_base, calc_controller
from model import calc_weak_form

if __name__ == '__main__':
    # initial conditions
    x0 = 0
    x_dt0 = 0

    # settings
    fem_order = 10
    exp_oder = 10

    params = pi.Parameters(sigma=1, tau=1, m=1,
                           alpha=1, kappa1=1, kappa0=1)
    spat_domain = pi.Domain(bounds=(0, 1), num=50)
    temp_dom = pi.Domain(bounds=(0, 10), num=50)

    # setup feedforward
    smooth_transition = pi.SmoothTransition((0, 1),
                                            (1, 3),
                                            method="poly",
                                            differential_order=2)
    ff = FlatFeedForward(smooth_transition, params)

    # setup controller
    exp_base, eig_vals = calc_exp_base(params, exp_oder, spat_domain)
    pi.register_base("exp_base", exp_base)

    pseudo_domain = pi.Domain(bounds=(-params.tau, params.tau), num=50)
    flat_base = calc_flat_base(eig_vals, pseudo_domain)
    pi.register_base("flat_base", flat_base)

    cont = calc_controller("exp_base", "flat_base", params)

    u = pi.SimulationInputSum([ff, cont])

    # setup system
    nodes, base = pi.cure_interval(pi.LagrangeFirstOrder,
                                   spat_domain.bounds,
                                   fem_order)
    pi.register_base("sim_base", base)

    system = calc_weak_form(params, "sim_base", spat_domain, u)

    # initial conditions
    ic = [
        pi.Function(lambda z: x0),  # x(z, 0)
        pi.Function(lambda z: x_dt0),  # x_dt(z, 0)
    ]

    # simulate system
    data = pi.simulate_system(system, ic, temp_dom, spat_domain)

    # collect data
    exp_weights = cont.get_results(data[0].input_data[0],
                                   result_key="exp_base")
    exp_data = pi.process_sim_data("exp_base",
                                   exp_weights,
                                   data[0].input_data[0],
                                   spat_domain, 0, 2,
                                   name="exp_approx")
    flat_data = pi.process_sim_data("flat_base",
                                    exp_weights,
                                    data[0].input_data[0],
                                    pseudo_domain, 0, 2,
                                    name="flat_approx")

    plot = pi.PgAnimatedPlot(data + exp_data + flat_data)
    pg.QAPP.exec_()
