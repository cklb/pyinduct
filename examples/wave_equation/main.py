import numpy as np

import pyinduct as pi
import pyqtgraph as pg

from feedforward import FlatFeedForward
from control import calc_exp_base, calc_flat_base, calc_controller, ExponentialStateBase
from model import calc_weak_form

if __name__ == '__main__':
    # initial conditions
    x0 = 0
    x_dt0 = 0.1

    # settings
    fem_order = 20
    exp_order = 2

    params = pi.Parameters(sigma=1, tau=.5, m=1,
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
    # sim_state = create_function_vectors("sim_base", zero_padding=True)
    # pi.register_base("sim_base_state", sim_state)

    exp_base, eig_vals = calc_exp_base(params, exp_order, spat_domain, debug=False)
    pi.register_base("exp_base", exp_base)

    exp_state = ExponentialStateBase("exp_base", eig_vals)
    pi.register_base("exp_base_state", exp_state)

    exp_state_dz = pi.Base([frac.get_member(0) for frac in exp_state.fractions])
    pi.register_base("e_dz", exp_state_dz)
    pi.visualize_functions(exp_state_dz.fractions)
    exp_state_dt = pi.Base([frac.get_member(1) for frac in exp_state.fractions])
    pi.register_base("e_dt", exp_state_dt)
    pi.visualize_functions(exp_state_dt.fractions)

    pseudo_domain = pi.Domain(bounds=(-params.tau, params.tau), num=50)
    flat_base = calc_flat_base(eig_vals, pseudo_domain, debug=False)
    pi.register_base("flat_base", flat_base)

    cont = calc_controller("exp_base_state", "flat_base", params)
    # cont = calc_controller("exp_base_state", "flat_base", params)

    u = pi.SimulationInputSum([ff, cont])

    # setup system

    system = calc_weak_form(params, "sim_base", spat_domain, u)

    # initial conditions
    ic = [
        pi.Function(lambda z: 0),  # x(z, 0)
        pi.Function(lambda z: x_dt0),  # x_dt(z, 0)
    ]

    # simulate system
    data = pi.simulate_system(system, ic, temp_dom, spat_domain,
                              derivative_orders=(1, 1))

    # collect data
    exp_weights = cont.get_results(data[0].input_data[0],
                                   result_key="exp_base_state"
                                   )
    exp_data = pi.process_sim_data("e_dt",
                                   exp_weights,
                                   pi.Domain(points=data[0].input_data[0]),
                                   spat_domain, 0, 0,
                                   name="exp_approx")
    flat_data = pi.process_sim_data("flat_base",
                                    exp_weights,
                                    pi.Domain(points=data[0].input_data[0]),
                                    pseudo_domain, 0, 2,
                                    name="flat_approx")
    for _data in exp_data:
        _data.output_data = np.real(_data.output_data)

    for _data in flat_data:
        _data.output_data = np.real(_data.output_data)

    plot = pi.PgAnimatedPlot((data
                              + exp_data
                              # + flat_data
                              ), replay_gain=1)

    pg.QAPP.exec_()
