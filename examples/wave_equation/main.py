import numpy as np

import pyinduct as pi
import pyqtgraph as pg

from feedforward import FlatFeedForward
from control import calc_exp_base, calc_flat_base, calc_controller, create_function_vectors
from model import calc_weak_form

if __name__ == '__main__':
    # initial conditions
    x0 = 10
    x_dt0 = 0

    # settings
    fem_order = 20
    exp_oder = 11

    params = pi.Parameters(sigma=1, tau=1, m=1,
                           alpha=1, kappa1=1, kappa0=1)
    spat_domain = pi.Domain(bounds=(0, 1), num=100)
    temp_dom = pi.Domain(bounds=(0, 10), num=100)

    # simulation base
    nodes, sim_base = pi.cure_interval(pi.LagrangeFirstOrder,
                                       spat_domain.bounds,
                                       fem_order)
    pi.register_base("sim_base", sim_base)

    # setup feedforward
    smooth_transition = pi.SmoothTransition((0, 1),
                                            (1, 3),
                                            method="poly",
                                            differential_order=2)
    ff = FlatFeedForward(smooth_transition, params)

    # setup controller
    sim_state = create_function_vectors("sim_base", zero_padding=True)
    pi.register_base("sim_base_state", sim_state)

    exp_base, eig_vals = calc_exp_base(params, exp_oder, spat_domain, debug=False)
    pi.register_base("exp_base", exp_base)

    exp_state = create_function_vectors("exp_base", eig_vals)
    pi.register_base("exp_base_state", exp_state)

    pseudo_domain = pi.Domain(bounds=(-params.tau, params.tau), num=50)
    flat_base = calc_flat_base(eig_vals, pseudo_domain, debug=False)
    pi.register_base("flat_base", flat_base)

    cont = calc_controller("exp_base", "flat_base", params)

    u = pi.SimulationInputSum([ff, cont])

    # setup system

    system = calc_weak_form(params, "sim_base", spat_domain, u)

    # initial conditions
    ic = [
        pi.Function(lambda z: z),  # x(z, 0)
        pi.Function(lambda z: x_dt0),  # x_dt(z, 0)
    ]

    # simulate system
    data = pi.simulate_system(system, ic, temp_dom, spat_domain,
                              derivative_orders=(0, 1))

    # collect data
    exp_weights = cont.get_results(data[0].input_data[0],
                                   # result_key="exp_base_state")
                                   result_key="exp_base")
    exp_data = pi.process_sim_data("exp_base",
                                   exp_weights,
                                   pi.Domain(points=data[0].input_data[0]),
                                   spat_domain, 0, 2,
                                   name="exp_approx")
    flat_data = pi.process_sim_data("flat_base",
                                    exp_weights,
                                    pi.Domain(points=data[0].input_data[0]),
                                    pseudo_domain, 0, 2,
                                    name="flat_approx")

    for _data in flat_data:
        _data.output_data = np.imag(_data.output_data)

    plot = pi.PgAnimatedPlot((data
                             # + exp_data
                              + flat_data
                              ))

    pg.QAPP.exec_()
