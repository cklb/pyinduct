import sympy as sp
import numpy as np
import pyinduct as pi
import pyinduct.sym_simulation as ss
from matplotlib import pyplot as plt

# approximation order
N = 3

temp_dom = pi.Domain((0, 10), num=100)

# spatial domain
spat_bounds = (0, 1)
spat_dom = pi.Domain(spat_bounds, num=10)

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
    # sp.Eq(sp.Subs(x, z, 0), u),
    sp.Eq(sp.Subs(x.diff(z), z, 0), u),
]

# define approximation basis
nodes = pi.Domain(spat_bounds, num=N)
fem_base = pi.LagrangeFirstOrder.cure_interval(nodes)
pi.register_base("fem", fem_base)

# create approximations, homogenizing where needed
x_approx = ss.create_approximation(z, "fem", boundaries)

# define initial conditions
x0 = sp.sin(z)
if 0:
    a = x_approx.get_spatial_approx(state0)
    vals = np.linspace(*spat_bounds)
    plt.plot(vals, np.sin(vals))
    plt.plot(vals, a(vals))
    plt.show()
    x0 = np.sin
    print(a(0))

# define the variational formulation for both phases
weak_form = [
    ss.InnerProduct(x.diff(t), phi_k, spat_bounds)
    - c * ((x*phi_k).subs(z, 1) - (x*phi_k).subs(z, 0)
           - ss.InnerProduct(x.diff(z), phi_k, spat_bounds))
]
sp.pprint(weak_form, num_columns=200)

x_test = x_approx.base
rep_dict = {
    x: x_approx,
    phi_k: x_test,
}

rep_eqs = ss.substitute_approximations(weak_form, rep_dict)
sp.pprint(rep_eqs, num_columns=200)

sys, state, inputs = ss.create_first_order_system(rep_eqs)
sp.pprint(inputs)
sp.pprint(state)
sp.pprint(sys, num_columns=200)


ic_dict = {
    x_approx: x0
}


def controller(t, weights):
    return weights[-1]


input_dict = {
    u: controller
}
np.seterr(under="warn")
res_weights = ss.simulate_state_space(temp_dom, sys, ic_dict, input_dict,
                                      inputs, state)

f, ax2 = plt.subplots(1, 1)
for idx, point in enumerate(res_weights.y):
    ax2.plot(res_weights.t, point, label="T_{}".format(idx))
ax2.legend()
ax2.grid()
plt.show()
