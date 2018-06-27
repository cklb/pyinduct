r"""
This example considers a two-phased stefan problem

- :math:`x_1(z,t)` ~ temperature distribution in the solid [x_1(z,t)] = K

- :math:`x_2(z,t)` ~ temperature distribution in the liquid [x_2(z,t)] = K

- :math:`\gamma(t)` ~ phase boundary position [\gamma(t)] = m

by the following equations:

.. math::
    :nowrap:

    \begin{align*}
    \end{align*}

"""

# (sphinx directive) start actual script
from pyinduct.tests import test_examples


if __name__ == "__main__" or test_examples:
    import sympy as sp
    from sympy.utilities.lambdify import lambdify, implemented_function
    import numpy as np
    from numbers import Number
    import pickle
    import pyinduct as pi

    class RampTrajectory(pi.SimulationInput):
        """
        Trajectory generator for a ramp as simulation input signal.

        Args:
            startValue (numbers.Number): Desired start value of the output.
            finalValue (numbers.Number): Desired value after step of the output
            stepStartTime (numbers.Number): Time where the ramp starts
            stepEndTime (numbers.Number): Time where the ramp ends
        """
        def __init__(self, startValue=0, finalValue=1, stepStartTime=1, stepEndTime=2, name=""):
            super().__init__(name)
            self._startValue = startValue
            self._finalValue = finalValue
            self._stepStartTime = stepStartTime
            self._stepEndTime = stepEndTime
            self._m = (self._finalValue - self._startValue) / (self._stepEndTime - self._stepStartTime)
            self._n = self._startValue - self._m * self._stepStartTime

        def _calc_output(self, **kwargs):
            if isinstance(kwargs["time"], (list, np.ndarray)):
                output = np.ones(len(np.atleast_1d(kwargs["time"]))) * self._startValue
                for idx, time in kwargs["time"]:
                    if self._stepStartTime <= time <= self._stepEndTime:
                        output[idx] = self._m * time + self._n
                    elif time < self._stepStartTime:
                        output[idx] = self._startValue
                    else:
                        output[idx] = self._finalValue
                return dict(output=output)
            elif isinstance(kwargs["time"], Number):
                if self._stepStartTime <= kwargs["time"] <= self._stepEndTime:
                    return dict(output=self._m * kwargs["time"] + self._n)
                elif kwargs["time"] < self._stepStartTime:
                    return dict(output=self._startValue)
                else:
                    return dict(output=self._finalValue)
            else:
                raise NotImplementedError

    def initial_condition_x1(z):
        return 0

    def initial_condition_x2(z):
        return 0

    def initial_condition_gamma(z):
        return 0


    # define some bases
    spat_dom = pi.Domain((0, 1), num=10)
    nodes, base = pi.cure_interval(pi.LagrangeFirstOrder,
                                   spat_dom.bounds,
                                   len(spat_dom))
    red_base = pi.Base(base[:-1])
    pi.register_base("fem_base_1", red_base)
    pi.register_base("fem_base_2", red_base)

    dirichlet_frac = base[-1]
    # pi.register_base("diri_base", dirichlet_frac)

    coll_fraction = pi.Function(lambda z: 1,
                                domain=spat_dom.bounds,
                                nonzero=spat_dom.bounds)
    coll_base = pi.Base([coll_fraction])
    pi.register_base("coll_base", coll_base)

    # parameters
    Tm = 0
    rho_m = 1  # kg/m^3
    L = 334  # kg/m^3
    rho = [0.91, 1]  # kg/m^3
    cp = [2.05, 4.19]
    k = [2.2, 0.591]
    alpha = [k[i]/(cp[i]*rho[i]) for i in range(2)]
    # interpolate = True

    # sympy symbols
    u1, u2, x1, x2, gamma, z, t = sp.symbols("u1 u2 x1 x2 gamma z t")
    x_sym = [x1(z, t), x2(z, t)]
    x_diri = implemented_function("x_d", dirichlet_frac)
    u_sym = [u1(t), u2(t)]

    # pyinduct placeholders
    u = pi.SimulationInputVector([RampTrajectory(startValue=0,
                                                 finalValue=6.2e-2,
                                                 stepStartTime=0.5,
                                                 stepEndTime=1),
                                  RampTrajectory(startValue=0,
                                                 finalValue=9e-2,
                                                 stepStartTime=3,
                                                 stepEndTime=4),
                                  ])
    sys_input = pi.Input(u)
    x_num = [pi.FieldVariable("fem_base_{}".format(i)) for i in range(1, 3)]
    gamma_num = pi.FieldVariable("coll_base")
    psi_num = pi.TestFunction("fem_base_1")
    psi_coll = pi.TestFunction("coll_base")

    # map sympy symbols to the corresponding base
    input_var_map = {0: u1(t), 1: u2(t)}
    base_var_map = {
        "fem_base_1": x1(z, t),
        "fem_base_2": x2(z, t),
        "col_base": gamma(z, t),
    }
    beta = [gamma(t) - spat_dom.bounds[i] for i in range(2)]
    scale2 = 1 / (rho_m * L)

    # weak formulations
    weak_forms = []
    for i in range(2):
        scale1 = alpha[i]/beta[i]**2
        form = pi.WeakFormulation([
            pi.IntegralTerm(pi.Product(x_num[i].derive(temp_order=1), psi_num),
                            limits=spat_dom.bounds, scale=-1),
            # part. integrated 2nd order term
            pi.SymbolicTerm(term=-scale1 * x_sym[i].diff(z),
                            test_function=psi_num.derive(1),
                            base_var_map=base_var_map,
                            input_var_map=input_var_map),
            pi.SymbolicTerm(term=scale1 * (-1)**i * beta[i] * u_sym[i] / k[i],
                            test_function=psi_num(1),
                            base_var_map=base_var_map,
                            input_var_map=input_var_map),
            pi.SymbolicTerm(term=-scale1 * x_sym[i].diff(z).subs(z, 0),
                            test_function=psi_num(0),
                            base_var_map=base_var_map,
                            input_var_map=input_var_map),
            # 1st order term
            pi.SymbolicTerm(term=scale2 * k[0] / beta[0] * x_sym[0].diff(z).subs(z, 1) * z * x_sym[i].diff(z),
                            test_function=psi_num,
                            base_var_map=base_var_map,
                            input_var_map=input_var_map),
            pi.SymbolicTerm(term=-scale2 * k[1] / beta[1] * x_sym[1].diff(z).subs(z, 1) * z * x_sym[i].diff(z),
                            test_function=psi_num,
                            base_var_map=base_var_map,
                            input_var_map=input_var_map),
        ], name="x_{}".format(i))
        weak_forms.append(form)

    wf_gamma = pi.WeakFormulation([
        pi.ScalarTerm(argument=gamma_num.derive(temp_order=1)(0)),
        pi.SymbolicTerm(term=scale2 * k[0]/beta[0] * x_sym[0].diff(z).subs(z, 1),
                        test_function=psi_coll(1),
                        base_var_map=base_var_map,
                        input_var_map=input_var_map),
        pi.SymbolicTerm(term=-scale2 * k[1]/beta[1] * x_sym[1].diff(z).subs(z, 1),
                        test_function=psi_coll(1),
                        base_var_map=base_var_map,
                        input_var_map=input_var_map),
    ], name="gamma")
    weak_forms.append(wf_gamma)

    # initial states
    ic_x1 = np.array([pi.Function(initial_condition_x1)])
    ic_x2 = np.array([pi.Function(initial_condition_x2)])
    ic_gamma = np.array([pi.Function(initial_condition_gamma)])

    ics = {
        weak_forms[0].name: ic_x1,
        weak_forms[1].name: ic_x2,
        weak_forms[2].name: ic_gamma
    }

    domains = {
        weak_forms[0].name: spat_dom,
        weak_forms[1].name: spat_dom,
        weak_forms[2].name: spat_dom
    }

    # simulation
    temp_domain = pi.Domain((0, 10), num=101)
    result = pi.simulate_systems(weak_forms, ics, temp_domain, domains)

    # visualization
    win = pi.PgAnimatedPlot(result)
    pi.show()

    # save results
    # pickle.dump(result, open('nonlin_tube_reactor.pkl', 'wb'))
    # nonlintube = pickle.load(open('nonlin_tube_reactor.pkl', 'rb'))
    # nonlintubefdm = pickle.load(open('nonlin_tube_reactor_fdm.pkl', 'rb'))
    # ps1 = PgAnimatedPlot([nonlintubefdm[0], nonlintube[3]])
    # ps2 = PgAnimatedPlot([nonlintubefdm[3], nonlintube[2]])
    # ps3 = PgAnimatedPlot([nonlintubefdm[1], nonlintube[0]])
    # ps4 = PgAnimatedPlot([nonlintubefdm[2], nonlintube[1]])
    # show()
