import pickle
import sympy as sp
import symengine as se
# from sympy.physics.quantum.innerproduct import InnerProduct

import pyinduct as pi
import pyinduct.sym_simulation as ss

# approximation order
N = 3

# spatial domain
spat_bounds = (0, 1)

# define temporal and spatial variables
t = ss.time
z = ss.space[0]
# z1, z2 = ss.space[:2]

# define system inputs
u1_t = ss.get_input()
u2_t = ss.get_input()

# define symbols of spatially distributed and lumped system variables
x1 = se.symbols("x1", cls=sp.Function)
x1_zt = x1(z, t)
x2 = se.symbols("x2", cls=sp.Function)
x2_zt = x2(z, t)
gamma = se.symbols("gamma", cls=sp.Function)
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
    (Gamma[0], spat_bounds[0]),
    (Gamma[1], spat_bounds[1]),
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
    se.Eq(se.Subs(x1_zt.diff(z), z, 0), -(gamma_t - Gamma[0]) / k[0] * u1_t),
    # se.Eq(se.Subs(x1_zt, z, 0), u1_t),
    se.Eq(se.Subs(x1_zt, z, 1), Tm),
]
boundaries_x2 = [
    se.Eq(se.Subs(x2_zt.diff(z), z, 0), (gamma_t - Gamma[1]) / k[1] * u2_t),
    # sp.Eq(se.Subs(x2_zt, z, 0), u2_t),
    se.Eq(se.Subs(x2_zt, z, 1), Tm),
]

if 0:
    # build state transformation
    zeta_trafo = [(z - Gamma[idx])/(gamma_t - Gamma[idx]) for idx in range(2)]
    x1_approx_z = x1_approx.subs(z, zeta_trafo[0])
    x2_approx_z = x2_approx.subs(z, zeta_trafo[1])
    print(x1_approx_z)
    quit()

# define approximation basis
nodes = pi.Domain(spat_bounds, num=N)
fem_base = pi.LagrangeFirstOrder.cure_interval(nodes)
pi.register_base("fem", fem_base)
# fem_base = ss.create_lag1ast_base(z, spat_bounds, N)

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
        ss.InnerProduct(x.diff(t), phi, spat_bounds)
        - alpha[idx] / beta**2 * (
            (x.diff(z) * phi).subs(z, 1)
            - (x.diff(z) * phi).subs(z, 0)
            - ss.InnerProduct(x.diff(z), phi.diff(z), spat_bounds)
        )
        - gamma_t.diff(t) / beta * ss.InnerProduct(z*x.diff(z), phi, spat_bounds)
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

# collect derivatives
targets = set()
for eq in rep_eqs:
    ders = eq.atoms(sp.Derivative)
    targets.update(ders)
targets = sorted(targets, key=lambda x: str(x))
print(targets)

if 0:
    # introduce definitional equations for higher order derivatives or derived
    # inputs
    extra_eqs = []
    extra_targets = []
    for t in targets:
        x = sp.Derivative()

# substitute all weights with new symbols
state_dict = dict()
state_dt_dict = dict()
state_derivatives = dict()
for t in targets:
    var = sp.Dummy()
    state_dict[t.args[0]] = var
    var_dt = sp.Dummy()
    state_dt_dict[t] = var_dt
    state_derivatives[var] = var_dt

print(state_dict)
print(state_dt_dict)
print(state_derivatives)

t_eqs = [eq.subs({**state_dict, **state_dt_dict}) for eq in rep_eqs]
print(t_eqs)

if 0:
    mat_form = sp.linear_eq_to_matrix(t_eqs, state_dt_dict.values())
    print(new_targets)
    print(mat_form)

print(">>> Solving for derivatives")
rhs = sp.solve(t_eqs, list(state_dt_dict.values()), dict=True)[0]
print(rhs)

# input and variables
all_vars = set()
for eq in rhs.values():
    sv = eq.atoms(sp.Function)
    all_vars.update(sv)

all_vars = sorted(all_vars, key=lambda _x: str(_x))
input_vars = list(filter(lambda var: "u" in str(var), all_vars))
state_vars = sorted(state_dict.values(), key=lambda _x: str(_x))
state = sp.Matrix(state_vars)
inputs = sp.Matrix(input_vars)
# print(state)
# print(inputs)

# build matrix expression for rhs
rhs_list = []
for var in state_vars:
    rhs_list.append(rhs[state_derivatives[var]])

rhs_vec = sp.Matrix(rhs_list)
jac = rhs_vec.jacobian(state)

# data = (inputs, state, rhs_vec)
data = (str(input_vars), str(state_vars), str(rhs_list), str(jac))
with open("symb_test_N={}.pkl".format(N), "wb") as f:
    pickle.dump(data, f)

