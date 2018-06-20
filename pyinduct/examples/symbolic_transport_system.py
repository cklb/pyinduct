import sympy as sp
# import symengine as sp
import numpy as np
import pyinduct as pi
import pyinduct.sym_simulation as ss
from matplotlib import pyplot as plt

# approximation order
N = 5

temp_dom = pi.Domain((0, 5), num=100)

# spatial domain
spat_bounds = (0, 1)
spat_dom = pi.Domain(spat_bounds, num=100)

# define temporal and spatial variables
t = ss.time
z = ss.space[0]

# define system inputs
u1 = ss.get_input()
u2 = ss.get_input()

# define symbols of spatially distributed and lumped system variables
x = ss.get_field_variable(z, t)

# define symbols for test functions
phi_k = ss.get_test_function(z)

# define parameters
alpha = sp.symbols("alpha", real=True)
param_list = [
    (alpha, .1),
    ("approx_pos", .5),
    ("approx_order", 5),
]
ss.register_parameters(*param_list)

# define boundaries
boundaries = [
    sp.Eq(sp.Subs(x.diff(z), z, spat_bounds[0], evaluate=False), -u1),
    sp.Eq(sp.Subs(x.diff(z), z, spat_bounds[1], evaluate=False), u2),
]

# define approximation basis
nodes = pi.Domain(spat_bounds, num=N)
fem_base = pi.LagrangeFirstOrder.cure_interval(nodes)
pi.register_base("fem", fem_base)

# create approximations, homogenizing where needed
x_approx = ss.create_approximation(z, "fem", boundaries)

# define initial conditions
x0 = 1e-1 * sp.sin(z*sp.pi)
# x0 = 10 * z
# x0 = lambda _z: .1

if 0:
    state0 = x_approx.approximate_function(x0)
    a = x_approx.get_spatial_approx(state0)
    vals = np.linspace(*spat_bounds)
    plt.plot(vals, a(vals))
    plt.show()

# define the variational formulation for both phases
# a0 = 0
# a0 = (1 + 10 * x)
a0 = x**2
# a0 = 2*x**2 + sp.exp(x)

weak_form = [
    ss.InnerProduct(x.diff(t), phi_k, spat_bounds)
    - alpha * ((x.diff(z)*phi_k).subs(z, 1)
               - (x.diff(z)*phi_k).subs(z, 0)
               - ss.InnerProduct(x.diff(z), phi_k.diff(z), spat_bounds))
    - ss.InnerProduct(a0 * x, phi_k, spat_bounds)  # some nasty scale for source
]
sp.pprint(weak_form, num_columns=200)

x_test = x_approx.base
rep_dict = {
    x: x_approx,
    phi_k: x_test,
}

rep_eqs = ss.substitute_approximations(weak_form, rep_dict)
# sp.pprint(rep_eqs, num_columns=200)

sys, state, inputs = ss.create_first_order_system(rep_eqs)
sp.pprint(inputs)
sp.pprint(state)
sp.pprint(sys, num_columns=200)

if 0:
    A, b = sp.linear_eq_to_matrix(sys, *state)
    A = np.array(A).astype(float)
    print(A)
    eigs = np.linalg.eigvals(A)
    print(eigs)

ic_dict = {
    x_approx: x0
}


def controller(t, weights):
    """ Top notch boundary feedback """
    k = 1e2
    # k = 0
    return -k * weights[0]


input_dict = {
    u1: controller,
    u2: controller
}
np.seterr(under="warn")
res_weights = ss.simulate_state_space(temp_dom, sys, ic_dict, input_dict,
                                      inputs, state)


t_dom = pi.Domain(points=res_weights.t)
weight_dict = ss._sort_weights(res_weights.y, state, [x_approx])
results = ss._evaluate_approximations(weight_dict, [x_approx], t_dom, spat_dom)

win = pi.PgAnimatedPlot(results)
pi.show()
