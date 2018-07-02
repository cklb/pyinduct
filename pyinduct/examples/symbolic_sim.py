import pickle
import numpy as np
import sympy as sp

import pyinduct as pi
import pyinduct.sym_simulation as ss

# approximation order
N = 3

# spatial domain
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
    (alpha[0], 1),
    (alpha[1], 2),
    (Gamma[0], zb_dom[0]),
    (Gamma[1], zb_dom[1]),
    (k[0], 1),
    (k[1], 2),
    (rho[0], 1),
    (rho[1], 2),
    (rho[2], 1.5),  # rho(Tm)
    (L, 1),
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


if 0:
    # build state transformation
    zeta_trafo = [(z - Gamma[idx])/(gamma_t - Gamma[idx]) for idx in range(2)]
    x1_approx_z = x1_approx.subs(z, zeta_trafo[0])
    x2_approx_z = x2_approx.subs(z, zeta_trafo[1])
    print(x1_approx_z)
    quit()

# define approximation basis
spat_dom = pi.Domain(zb_dom, num=N)
if 1:
    fem_base = pi.LagrangeFirstOrder.cure_interval(spat_dom)
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
            - (x.diff(z) * phi).subs(z, 1)
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

sys, state, inputs = ss.create_first_order_system(rep_eqs)
sp.pprint(inputs)
sp.pprint(state)
sp.pprint(sys, num_columns=200)

if 0:
    data = str(inputs), str(state), str(sys)
    with open("symb_test_N={}.pkl".format(N), "wb") as f:
        pickle.dump(data, f)
    quit()

# define the initial conditions for each approximation
ic_dict = {
    x1_approx: -10 + 10*z,
    x2_approx: 10 - 10*z,
    gamma_approx: .5,
}

# define the system inputs and their mapping

def controller_factory(idx, gain):

    def control_law(t, weights):
        """ Top notch boundary feedback """
        return -gain * weights[idx]

    return control_law


input_dict = {
    u1_t: controller_factory(0, 1),
    u2_t: controller_factory(-1, 1),
}

# run the simulation
np.seterr(under="warn")
res_weights = ss.simulate_state_space(temp_dom, sys, ic_dict, input_dict,
                                      inputs, state)


t_dom = pi.Domain(points=res_weights.t)
approximations = [x1_approx, x2_approx]
weight_dict = ss._sort_weights(res_weights.y, state, approximations)
results = ss._evaluate_approximations(weight_dict, approximations, t_dom, spat_dom)

win = pi.PgAnimatedPlot(results)
win2 = pi.PgSurfacePlot(results[0])
pi.show()


