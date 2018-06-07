import pickle
import numpy as np
import sympy as sp
from sympy.utilities.autowrap import ufuncify

from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

N = 3

with open("symb_test_N={}.pkl".format(N), "rb") as f:
    data = pickle.load(f)

inputs, state, rhs_vec, rhs_jac = sp.sympify(data)

print(inputs)
print(state)
print(rhs_vec)

rhs_cb = sp.lambdify(args=[inputs, state], expr=rhs_vec, modules="numpy")
jac_cb = sp.lambdify(args=[inputs, state], expr=rhs_jac, modules="numpy")
# cb = ufuncify([inputs, state], rhs_vec, backend="numpy")

# q_dt = cb(np.array([0, 0]), np.array([0, 0, 0, 0, .5]))
# q_dt = cb(np.array([0, 0]), np.array([-1, -1, 1, 1, .5]))
# print(q_dt)


def inputs(t):
    return np.array([-10, 0])


def rhs(t, y):
    u = inputs(t)
    y_dt = rhs_cb(u, y)
    return y_dt


def jac(t, y):
    u = inputs(t)
    jac = jac_cb(u, y)
    return jac


y0 = np.array([0, 0, 0, 0, .1])
res = solve_ivp(rhs, (0, 100), y0, jac=jac, method="BDF")

f = plt.figure()
for point in res.y:
    plt.plot(res.t, point)
plt.show()
