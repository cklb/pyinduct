import pyinduct as pi


class FlatFeedForward(pi.SimulationInput):
    def __init__(self, desired_handle, params):
        super().__init__()
        self._y = desired_handle
        self._params = params

    def _calc_output(self, **kwargs):
        y_p = self._y(kwargs["time"] + self._params.tau)
        y_m = self._y(kwargs["time"] - self._params.tau)
        f = + self._params.kappa0 * (y_p[0] + self._params.alpha * y_m[0]) \
            + self._params.kappa1 * (y_p[1] + self._params.alpha * y_m[1]) \
            + y_p[2] + self._params.alpha * y_m[2]
        return dict(output=self._params.m / 2 * f)
