import logging
import pickle
import dill
import numpy as np
import sympy as sp

import pyinduct as pi
import pyinduct.sym_simulation as ss

import pyqtgraph as pg

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

complete_dom = (0, 2)

# approximation order
N = 20

# split, transformed, constant domain
spat_dom = pi.Domain((0, 1), num=N)
zeta = sp.symbols("zeta:2")

# temporal domain
temp_dom = pi.Domain((0, 5), num=100)

# define temporal and spatial variables
t = ss.time
z = ss.space[0]
# z1, z2 = ss.space[:2]

# define system inputs
u1 = ss.get_input()
u2 = ss.get_input()

# define symbols of spatially distributed and lumped system variables
x1 = ss.get_field_variable(z, t)
x2 = ss.get_field_variable(z, t)
gamma = ss.get_field_variable(t)
# gamma = sp.symbols("gamma", cls=sp.Function)(t)
# gamma = gamma.subs(z, 1)

# define symbols for test functions
phi1_k = ss.get_test_function(z)
phi2_k = ss.get_test_function(z)

# define parameters
alpha, Gamma, k, rho, L, Tm = sp.symbols(("alpha:2",
                                          "Gamma:2",
                                          "k:2",
                                          "rho:3",
                                          "L",
                                          "T_m"), real=True)
param_list = [
    (alpha[0], .591/(2.05*.91)),
    (alpha[1], 2.2/(4.19*1)),
    (Gamma[0], complete_dom[0]),
    (Gamma[1], complete_dom[1]),
    (k[0], .591),
    (k[1], 2.2),
    (rho[0], .91),
    (rho[1], 1),
    (rho[2], 1),  # rho(Tm)
    (L, 334e3),
    (Tm, 0),
]
ss.register_parameters(*param_list)

# define original system
zeta_trafo = [(z - Gamma[idx])/(gamma - Gamma[idx]) for idx in range(2)]

# define boundaries
boundaries_x1 = [
    sp.Eq(sp.Subs(x1.diff(z), z, 0), -(gamma - Gamma[0]) / k[0] * u1),
    sp.Eq(sp.Subs(x1, z, 1), Tm),
]
boundaries_x2 = [
    sp.Eq(sp.Subs(x2.diff(z), z, 0), (gamma - Gamma[1]) / k[1] * u2),
    sp.Eq(sp.Subs(x2, z, 1), Tm),
]

# define approximation basis
if 1:
    nodes = pi.Domain(spat_dom.bounds, num=N)
    fem_base = pi.LagrangeFirstOrder.cure_interval(nodes)
else:
    fem_base = ss.create_lag1ast_base(z, zb_dom, N)
pi.register_base("fem", fem_base)

# create approximations, homogenizing where needed
x1_approx = ss.create_approximation(z, "fem", boundaries_x1)
x2_approx = ss.create_approximation(z, "fem", boundaries_x2)
# gamma_approx = ss.create_approximation(z, "gamma")
gamma_approx = ss.get_weight()

if 0:
    # TODO coord trafo
    x_part_1 = x1_approx.expression.subs(z, zeta_trafo[0])
    x_part_2 = x2_approx.expression.subs(z, zeta_trafo[1])
    x_expr = sp.Piecewise((x_part_1, z < gamma),
                          # (Tm, z == gamm),
                          (x_part_2, z > gamma))
    x_approx = ss.LumpedApproximation()

# define the initial conditions for each approximation
ics = {
    x1_approx: -10 * (1-z),
    x2_approx: 10 * (1-z),
    gamma_approx: .5
}

# define the system inputs and their mapping
input_map = {
    u1: pi.ConstantTrajectory(-500),
    u2: pi.ConstantTrajectory(500),
}

# define the variational formulation for both phases
equations = []
for idx, (x, u, phi) in enumerate(zip([x1, x2],
                                      [u1, u2],
                                      [phi1_k, phi2_k])):
    beta = (gamma - Gamma[idx])
    expr = (
        ss.InnerProduct(x.diff(t), phi, spat_dom.bounds)
        - alpha[idx] / beta ** 2 * (
            (x.diff(z) * phi).subs(z, 1)
            - (x.diff(z) * phi).subs(z, 0)
            - ss.InnerProduct(x.diff(z), phi.diff(z), spat_dom.bounds)
        )
        - gamma.diff(t) / beta * ss.InnerProduct(z * x.diff(z), phi, spat_dom.bounds)
    )
    equations.append(expr)

# define the ode for the phase boundary
g_exp = (k[0] / (gamma - Gamma[0]) * x1.diff(z).subs(z, 1)
         - k[1] / (gamma - Gamma[1]) * x2.diff(z).subs(z, 1)
         - gamma.diff(t) * rho[2] * L
         )
equations.append(g_exp)
sp.pprint(equations)

# create test functions, easy due to Galerkin principle
x1_test = x1_approx.base
x2_test = x2_approx.base

approx_map = {
    x1: x1_approx,
    x2: x2_approx,
    phi1_k: x1_test,
    phi2_k: x2_test,
    gamma: gamma_approx
}

if 0:
    # build complete form
    rep_eqs, approximations = ss.substitute_approximations(equations, approx_map)

    # convert to state space system
    ss_sys = ss.create_first_order_system(rep_eqs, input_map)

    f_name = "tpsp_N={}.dl".format(N)
    ss_sys.dump(f_name)
    ss_sys = ss.SymStateSpace.from_file(f_name)

    # process initial conditions
    y0 = ss.calc_initial_sate(ss_sys, ics, temp_dom[0])
else:
    results = ss.simulate_system(equations, approx_map, input_map, ics,
                                 temp_dom, spat_dom,
                                 extra_derivatives=[(z, 1)])

if 0:
    data = str(inputs), str(state), str(sys)
    with open("symb_test_N={}.pkl".format(N), "wb") as f:
        pickle.dump(data, f)
    quit()

# post processing

if 0:
    # build state transformation
    zeta_trafo = [(z - Gamma[idx])/(gamma_approx - Gamma[idx]) for idx in range(2)]
    # approximations = [appr.subs(z, trafo) for appr, trafo in
    #                   zip([x1_approx, x2_approx], zeta_trafo)]
else:
    approximations = [x1_approx, x2_approx]

# gamma_sim = res_weights.y[-1]
plots = []

# initial conditions
# for res in results:
#     p = pg.plot(res.input_data[1].points, res.output_data[0, :])
#     plots.append(p)

res = results[tuple()]
res_dz = results[(z, 1)]
res_gamma = res.pop(-1)
res_gamma_dz = res_dz.pop(-1)
p = pg.plot(res_gamma.input_data[0].points, res_gamma.output_data)
p.plot(res_gamma_dz.input_data[0].points, res_gamma_dz.output_data)

win = pi.PgAnimatedPlot(res + res_dz)
# win1 = pi.PgSurfacePlot(results[0])
# win2 = pi.PgSurfacePlot(results[1])
pi.show()


