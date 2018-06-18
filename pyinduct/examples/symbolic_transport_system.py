import sympy as sp
import pyinduct as pi
import pyinduct.sym_simulation as ss

# approximation order
N = 3

# spatial domain
spat_bounds = (0, 1)

# define temporal and spatial variables
t = ss.time
z = ss.space[0]

# define system inputs
u = ss.get_input()

# define symbols of spatially distributed and lumped system variables
x = ss.get_field_variable(z, t)

# define symbols for test functions
phi_k = ss.get_test_function(z)

# define parameters
c = sp.symbols("c", real=True)
param_list = [(c, 1)]
ss.register_parameters(*param_list)

# define boundaries
boundaries = [
    sp.Eq(sp.Subs(x.diff(z), z, 0), u),
]

# define approximation basis
nodes = pi.Domain(spat_bounds, num=N)
fem_base = pi.LagrangeFirstOrder.cure_interval(nodes)
pi.register_base("fem", fem_base)

# create approximations, homogenizing where needed
x_approx = ss.create_approximation(z, "fem", boundaries)

# define the variational formulation for both phases
weak_form = [
    ss.InnerProduct(x.diff(t), phi_k, spat_bounds)
    - c * ss.InnerProduct(x.diff(z), phi_k, spat_bounds)
]
sp.pprint(weak_form, num_columns=200)

x_test = x_approx.base
rep_dict = {
    x: x_approx,
    phi_k: x_test,
}

rep_eqs = ss.substitute_approximations(weak_form, rep_dict)
# sp.pprint(rep_eqs)

sys, maps = ss.create_first_order_system(rep_eqs)
sp.pprint(sys, num_columns=200)
