import pickle
import numpy as np
import sympy as sp

import pyinduct as pi
import pyinduct.sym_simulation as ss

import pyqtgraph as pg

# approximation order
N = 3

# spatial domains
spat_dom = pi.Domain((0, .1), num=N)
zb_dom = (0, 1)

# temporal domain
temp_dom = pi.Domain((0, 5), num=100)

# define temporal and spatial variables
t = ss.time
z = ss.space[0]
# z1, z2 = ss.space[:2]

# define system inputs
u1_t = ss.get_input()
u2_t = ss.get_input()

# define symbols of spatially distributed and lumped system variables
x1_zt = ss.get_field_variable(z, t)
x2_zt = ss.get_field_variable(z, t)
gamma = sp.symbols("gamma", cls=sp.Function)
gamma_t = gamma(t)

# define symbols for test functions
phi_1kz = ss.get_test_function(z)
phi_2kz = ss.get_test_function(z)

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
    (Gamma[0], zb_dom[0]),
    (Gamma[1], zb_dom[1]),
    (k[0], .591),
    (k[1], 2.2),
    (rho[0], .91),
    (rho[1], 1),
    (rho[2], 1),  # rho(Tm)
    (L, 334e3),
    (Tm, 0),
]
ss.register_parameters(*param_list)

# define boundaries
boundaries_x1 = [
    sp.Eq(sp.Subs(x1_zt.diff(z), z, 0), -(gamma_t - Gamma[0]) / k[0] * u1_t),
    # se.Eq(se.Subs(x, z, 0), u1_t),
    sp.Eq(sp.Subs(x1_zt, z, 1), Tm),
]
boundaries_x2 = [
    sp.Eq(sp.Subs(x2_zt.diff(z), z, 0), (gamma_t - Gamma[1]) / k[1] * u2_t),
    # sp.Eq(se.Subs(x2_zt, z, 0), u2_t),
    sp.Eq(sp.Subs(x2_zt, z, 1), Tm),
]

# define approximation basis
if 1:
    nodes = pi.Domain(zb_dom, num=N)
    fem_base = pi.LagrangeFirstOrder.cure_interval(nodes)
else:
    fem_base = ss.create_lag1ast_base(z, zb_dom, N)
pi.register_base("fem", fem_base)

# create approximations, homogenizing where needed
x1_approx = ss.create_approximation(z, "fem", boundaries_x1)
x2_approx = ss.create_approximation(z, "fem", boundaries_x2)
gamma_approx = ss.get_weight()


# define the variational formulation for both phases
equations = []
for idx, (x, u, phi) in enumerate(zip([x1_zt, x2_zt],
                                      [u1_t, u2_t],
                                      [phi_1kz, phi_2kz])):
    beta = (gamma_t - Gamma[idx])
    expr = (
        ss.InnerProduct(x.diff(t), phi, zb_dom)
        - alpha[idx] / beta**2 * (
            (x.diff(z) * phi).subs(z, 1)
            - (x.diff(z) * phi).subs(z, 0)
            - ss.InnerProduct(x.diff(z), phi.diff(z), zb_dom)
        )
        - gamma_t.diff(t) / beta * ss.InnerProduct(z * x.diff(z), phi, zb_dom)
    )
    equations.append(expr)

# define the ode for the phase boundary
g_exp = (k[0] / (gamma_t - Gamma[0]) * x1_zt.diff(z).subs(z, 1)
         - k[1] / (gamma_t - Gamma[1]) * x2_zt.diff(z).subs(z, 1)
         - gamma_t.diff(t) * rho[2] * L
         )
equations.append(g_exp)
sp.pprint(equations)

# print(u1_t in equations[0].atoms(sp.Function))
# print(equations[0])

# print(x1_approx)
# print(x1_approx)
# print(x1_approx.atoms(sp.Function))
# print(gamma_approx.atoms(sp.Function))
# x1_list.append((x1, lambda _z, _t: x1_approx.subs([(z, _z), (t, _t)])))

# create test functions, easy due to Galerkin principle
x1_test = x1_approx.base
x2_test = x2_approx.base

rep_dict = {
    x1_zt: x1_approx,
    x2_zt: x2_approx,
    phi_1kz: x1_test,
    phi_2kz: x2_test,
    gamma_t: gamma_approx
}

rep_eqs = ss.substitute_approximations(equations, rep_dict)
print(rep_eqs)

ss_sys = ss.create_first_order_system(rep_eqs)
# sp.pprint(ss_sys.rhs, num_columns=200)

# define the initial conditions for each approximation
ss_sys.ics = {
    x1_approx: lambda z: -10,
    # x1_approx: -10 + 10*z,
    x2_approx: lambda z: 10,
    # x2_approx: 10 - 10*z,
    gamma_approx: .5,
}

# define the system inputs and their mapping
def controller_factory(idx, gain):

    def control_law(**kwargs):
        """ Top notch boundary feedback """
        return -gain * kwargs["weights"][idx]

    return control_law

def feedfoward_factory(val):

    def ff_law(**kwargs):
        """ Top notch boundary feedback """
        return val

    return ff_law


ss_sys.input_map = {
    u1_t: feedfoward_factory(-500),
    u2_t: feedfoward_factory(0),
    # u1_t: controller_factory(0, 1),
    # u2_t: controller_factory(-1, 1),
}

if 0:
    data = str(inputs), str(state), str(sys)
    with open("symb_test_N={}.pkl".format(N), "wb") as f:
        pickle.dump(data, f)
    quit()

# run the simulation
np.seterr(under="warn")
res_weights = ss.simulate_state_space(ss_sys, temp_dom)
t_dom = pi.Domain(points=res_weights.t)

# post processing

if 0:
    # build state transformation
    zeta_trafo = [(z - Gamma[idx])/(gamma_approx - Gamma[idx]) for idx in range(2)]
    # approximations = [appr.subs(z, trafo) for appr, trafo in
    #                   zip([x1_approx, x2_approx], zeta_trafo)]
else:
    approximations = [x1_approx, x2_approx]

weight_dict = ss._sort_weights(res_weights.y, ss_sys.state, approximations)
results = ss._evaluate_approximations(weight_dict, approximations, t_dom, nodes)
gamma_sim = res_weights.y[-1]
pg.plot(t_dom.points, gamma_sim)

win = pi.PgAnimatedPlot(results)
win1 = pi.PgSurfacePlot(results[0])
win2 = pi.PgSurfacePlot(results[1])
pi.show()


