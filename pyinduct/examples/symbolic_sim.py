import pickle
import numpy as np
import pyinduct as pi
import pyinduct.sym_simulation as ss
import sympy as sp
# from sympy.physics.quantum.innerproduct import InnerProduct

import matplotlib.pyplot as plt

# approximation order
N = 50
spat_bounds = (0, 1)

# define temporal and spatial variables
t = ss.time
z = ss.space[0]
# z1, z2 = ss.space[:2]

# define symbols of spatially distributed and lumped system variables
x1 = sp.symbols("x1", cls=sp.Function)
x1_zt = x1(z, t)
x2 = sp.symbols("x2", cls=sp.Function)
x2_zt = x2(z, t)
gamma = sp.symbols("gamma", cls=sp.Function)
gamma_t = gamma(t)

# define symbols for test functions
phi = sp.symbols("phi_(:{})".format(N-1), cls=sp.Function)
phi_z = [p(z) for p in phi]
print(phi)
print(phi_z)

# define system inputs
u1 = sp.symbols("u1", cls=sp.Function)
u1_t = u1(t)
u2 = sp.symbols("u2", cls=sp.Function)
u2_t = u2(t)

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

equations = []

# define the variational formulation for both phases
for idx, (x, u) in enumerate(zip([x1_zt, x2_zt], [u1_t, u2_t])):
    weak_forms = []
    for p in phi_z:
        beta = (gamma_t - Gamma[idx])
        delta = -(-1) ** idx
        exp = (
            ss.InnerProduct(x.diff(t), p, spat_bounds)
            - alpha[idx] / beta**2 * (
                (x.diff(z) * p).subs(z, 1)
                - delta * beta / k[idx] * u * p.subs(z, 0)
                - ss.InnerProduct(x.diff(z), p.diff(z), spat_bounds)
            )
            - gamma_t.diff(t) / beta * ss.InnerProduct(z*x.diff(z), p, spat_bounds)
        )
        # sp.pprint(exp)
        # quit()
        weak_forms.append(exp)
    equations += weak_forms

# define the ode for the phase boundary
g_exp = (k[0] / (gamma_t - Gamma[0]) * x1_zt.diff(z).subs(z, 1)
         - k[1] / (gamma_t - Gamma[1]) * x2_zt.diff(z).subs(z, 1)
         - gamma_t.diff(t) * rho[2] * L
         )
equations.append(g_exp)
# print(u1_t in equations[0].atoms(sp.Function))
# print(equations[0])

# approximations
fem_base = ss.create_lag1ast_base(z, spat_bounds, N)
x1_approx = ss.create_approximation(z, fem_base, ess_bounds=[(1, Tm)])
x2_approx = ss.create_approximation(z, fem_base, ess_bounds=[(1, Tm)])
gamma_approx = ss.get_weights(1)[0]
# print(x1_approx)
# print(x1_approx)
# print(x1_approx.atoms(sp.Function))
# print(gamma_approx.atoms(sp.Function))

# specify replacement dicts
variable_list = [
    (x1, lambda _z, _t: x1_approx.subs([(z, _z), (t, _t)])),
    (x2, lambda _z, _t: x2_approx.subs([(z, _z), (t, _t)])),
    (gamma, lambda _t: gamma_approx.subs(t, _t)),
]


def wrap_expr(expr, sym):
    def wrapped_func(_z):
        return expr.subs(sym, _z)

    return wrapped_func


test_list = [(p, wrap_expr(f, z))
             for p, f in zip(phi, fem_base[:-1])]
print(test_list)

rep_eqs = []
for eq in equations:
    rep_eq = eq
    for pair in variable_list + test_list:
        rep_eq = rep_eq.replace(*pair)

    rep_eqs.append(rep_eq.subs(param_list).doit())

# print(u1_t in rep_eqs[0].atoms(sp.Function))
# print(rep_eqs[0])
# quit()

# collect derivatives
targets = set()
for eq in rep_eqs:
    ders = eq.atoms(sp.Derivative)
    targets.update(ders)
targets = sorted(targets, key=lambda x: str(x))
print(targets)

if 0:
    # introduce definitional equations for higher order derivatives
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
    mat_form = sp.linear_eq_to_matrix(t_eqs, list(state_dt_dict.values))
    print(new_targets)
    print(mat_form)

rhs = sp.solve(t_eqs, list(state_dt_dict.values()))
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
