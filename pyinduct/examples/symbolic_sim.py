import pyinduct as pi
import pyinduct.sym_simulation as ss
import sympy as sp
# from sympy.physics.quantum.innerproduct import InnerProduct

# approximation order
N = 10

# define temporal and spatial variables
t = ss.time
z = ss.space[0]
# z1, z2 = ss.space[:2]

# define symbols of spatially distributed and lumped system variables
x1 = sp.symbols("x1", cls=sp.Function)
x2 = sp.symbols("x1", cls=sp.Function)
gamma = sp.symbols("gamma", cls=sp.Function)

# define symbols for test functions
# phi = sp.symbols("phi1", cls=sp.Function)
phi = sp.symbols("phi1(:{})".format(N), cls=sp.Function)
print(phi)

# define system inputs
u1 = sp.symbols("u1", cls=sp.Function)
u2 = sp.symbols("u2", cls=sp.Function)

# define parameters
alpha, Gamma, k, rho, L = sp.symbols(("alpha:2",
                                      "Gamma:2",
                                      "k:2",
                                      "rho:2",
                                      "L"))

equations = []

# define the variational formulation for both phases
for idx, (x, u) in enumerate(zip([x1, x2], [u1, u2])):
    weak_forms = []
    for p in phi:
        beta = (gamma(t) - Gamma[idx])
        exp = (
            ss.InnerProduct(x(z, t).diff(t), p(z))
            - alpha[idx] / beta**2 * (
                (x(z, t).diff(z) * p(z)).subs(z, 1)
                - beta / k[idx] * u(t) * p(0)
                - ss.InnerProduct(x(z, t).diff(z), p(z).diff(z))
            )
            - gamma(t).diff(t) / beta * ss.InnerProduct(z*x1(z, t).diff(z),
                                                        p(z))
        )
        # sp.pprint(exp)
        # quit()
        weak_forms.append(exp)
    equations += weak_forms

# define the ode for the phase boundary
g_exp = 1/(rho[1]*L)*(
    k[0]/(gamma(t) - Gamma[0]) * x1(z, t).diff(z).subs(z, 1)
    - k[1]/(gamma(t) - Gamma[1]) * x2(z, t).diff(z).subs(z, 1)
)
# sp.pprint(g_exp)
equations.append(g_exp)

# define approximations

