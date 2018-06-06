import numpy as np
import pyinduct as pi
import pyinduct.sym_simulation as ss
import sympy as sp
# from sympy.physics.quantum.innerproduct import InnerProduct

import matplotlib.pyplot as plt

# approximation order
N = 3
spat_bounds = (0, 1)

# define temporal and spatial variables
t = ss.time
z = ss.space[0]
# z1, z2 = ss.space[:2]

# define symbols of spatially distributed and lumped system variables
x1 = sp.symbols("x1", cls=sp.Function)(z, t)
x2 = sp.symbols("x1", cls=sp.Function)(z, t)
gamma = sp.symbols("gamma", cls=sp.Function)(t)

# define symbols for test functions
# phi = sp.symbols("phi1", cls=sp.Function)
phi = [sp.symbols("phi_{}".format(i), cls=sp.Symbol)(z)
       for i in range(N)]
print(phi)

# define system inputs
u1 = sp.symbols("u1", cls=sp.Function)
u2 = sp.symbols("u2", cls=sp.Function)

# define parameters
alpha, Gamma, k, rho, L, Tm = sp.symbols(("alpha:2",
                                          "Gamma:2",
                                          "k:2",
                                          "rho:2",
                                          "L",
                                          "T_m"), real=True)

equations = []

# define the variational formulation for both phases
for idx, (x, u) in enumerate(zip([x1, x2], [u1, u2])):
    weak_forms = []
    for p in phi:
        beta = (gamma - Gamma[idx])
        exp = (
            ss.InnerProduct(x.diff(t), p, spat_bounds)
            - alpha[idx] / beta**2 * (
                (x.diff(z) * p).subs(z, 1)
                - beta / k[idx] * u(t) * p.subs(z, 0)
                - ss.InnerProduct(x.diff(z), p.diff(z), spat_bounds)
            )
            - gamma.diff(t) / beta * ss.InnerProduct(z*x.diff(z), p, spat_bounds)
        )
        # sp.pprint(exp)
        # quit()
        weak_forms.append(exp)
    equations += weak_forms

# define the ode for the phase boundary
g_exp = 1/(rho[1]*L)*(
    k[0]/(gamma - Gamma[0]) * x1.diff(z).subs(z, 1)
    - k[1]/(gamma - Gamma[1]) * x2.diff(z).subs(z, 1)
)
# sp.pprint(g_exp)
equations.append(g_exp)

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
subs_list = [(x1, x1_approx),
             (x2, x2_approx),
             (gamma, gamma_approx),
             ] + list(zip(phi, fem_base))
s_eqs = [eq.subs(subs_list).doit() for eq in equations]
print([eq for eq in s_eqs])

# aquire targets
targets = set()
for eq in s_eqs:
    targets.update(eq.atoms(sp.Derivative))

print(targets)

res = sp.solve(s_eqs, targets, dict=True)
print(res)


