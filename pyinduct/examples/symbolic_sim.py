import time
import pickle
import sympy as sp
# from sympy.physics.quantum.innerproduct import InnerProduct
from tqdm import tqdm

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
input_types = ["dirichlet", "dirichlet"]
u1 = sp.symbols("u1", cls=sp.Function)
u1_t = u1(t)
u2 = sp.symbols("u2", cls=sp.Function)
u2_t = u2(t)

# define symbols of spatially distributed and lumped system variables
x1 = sp.symbols("x1", cls=sp.Function)
x1_zt = x1(z, t)
x2 = sp.symbols("x2", cls=sp.Function)
x2_zt = x2(z, t)
gamma = sp.symbols("gamma", cls=sp.Function)
gamma_t = gamma(t)

# define symbols for test functions
phi_1k = sp.symbols("phi1_k", cls=sp.Function)
phi_1kz = phi_1k
phi_2k = sp.symbols("phi2_k", cls=sp.Function)
phi_2kz = phi_2k

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
    sp.Eq(sp.Subs(x1_zt.diff(z), z, 0), -(gamma_t - Gamma[0]) / k[0] * u1_t),
    # sp.Eq(sp.Subs(x1_zt, z, 0), u1_t),
    sp.Eq(sp.Subs(x1_zt, z, 1), Tm),
]
boundaries_x2 = [
    sp.Eq(sp.Subs(x2_zt.diff(z), z, 0), (gamma_t - Gamma[1]) / k[1] * u2_t),
    # sp.Eq(sp.Subs(x2_zt, z, 0), u2_t),
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
fem_base = ss.create_lag1ast_base(z, spat_bounds, N)

# create approximations, homogenizing where needed
x1_approx, x1_test = ss.create_approximation(z, fem_base, boundaries_x1)
x2_approx, x2_test = ss.create_approximation(z, fem_base, boundaries_x2)
gamma_approx = ss.get_weights(1)[0]


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
# print(u1_t in equations[0].atoms(sp.Function))
# print(equations[0])

# print(x1_approx)
# print(x1_approx)
# print(x1_approx.atoms(sp.Function))
# print(gamma_approx.atoms(sp.Function))
# x1_list.append((x1, lambda _z, _t: x1_approx.subs([(z, _z), (t, _t)])))


def wrap_expr(expr, *sym):
    def wrapped_func(*_sym):
        return expr.subs(list(zip(sym, _sym)))

    return wrapped_func


def gen_func_subs_pair(func, expr):
    if isinstance(func, sp.Function):
        a = func.func
        args = func.args
    elif isinstance(func, sp.Subs):
        a = func
        if isinstance(func.args[0], sp.Derivative):
            args = func.args[0].args[0].args
        elif isinstance(func.args[0], sp.Function):
            args = func.args[0].args
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    b = wrap_expr(expr, *args)
    return a, b


# specify replacement dicts
x1_list = [
    gen_func_subs_pair(*c.args) for c in boundaries_x1
]
x1_list.append(gen_func_subs_pair(x1_zt, x1_approx))
x2_list = [
    gen_func_subs_pair(*c.args) for c in boundaries_x2
]
x2_list.append(gen_func_subs_pair(x2_zt, x2_approx))
test_list_1 = [gen_func_subs_pair(*p) for p in zip(phi_1z, x1_test)]
test_list_2 = [gen_func_subs_pair(*p) for p in zip(phi_2z, x2_test)]
gamma_list = [gen_func_subs_pair(gamma_t, gamma_approx)]

variable_list = x1_list + x2_list + test_list_1 + test_list_2 + gamma_list
# print(variable_list)

# substitute formulations
t0 = time.clock()
rep_eqs = []
for eq in tqdm(equations):
    rep_eq = eq
    # rep_eq = eq.subs(param_list)
    # print(rep_eq)
    for pair in tqdm(variable_list):
        # print("Substituting pair:")
        # print(pair)
        rep_eq = rep_eq.replace(*pair)
        # if u1_t in rep_eq.atoms(sp.Function):
            # print(rep_eq.atoms(sp.Derivative))
        # print("Result:")
        # print(rep_eq)

    rep_eqs.append(rep_eq.subs(param_list).doit())
print(time.clock() - t0)
# quit()

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

