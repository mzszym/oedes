# -*- coding: utf-8; -*-
#
# oedes - organic electronic device simulator
# Copyright (C) 2017-2018 Marek Zdzislaw Szymanski (marek@marekszymanski.com)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License, version 3,
# as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

from . import solver
from .model import model
from .models import Poisson
from .models.solver import PoissonOnly, SolverObject

from collections import namedtuple
import numpy as np
import bisect

contextstate = namedtuple('contextstate', ['x', 'xt', 'time', 'params'])

import tqdm

progressbar_type = tqdm.tqdm


class solution_vector(object):
    "Solution vector with metadata: model, time, params, and references to solutions in previous timesteps for transient simulation"

    def __init__(self, model, time, x, params, old=(), weights=None):
        self.model = model
        self.time = time
        self.x = x
        self.old = old
        self.params = params
        self.weights = weights

    def xt(self):
        "Numerically evaluate xt using weights"
        return self.weights[0] * self.x + \
            sum(o.x * w for o, w in zip(self.old, self.weights[1:]))

    def interpolate(self, time):
        "Return interpolated solution vector"
        raise NotImplementedError()


class steadystate_solution(solution_vector):
    "Steady-state solution"

    def __init__(self, model, time, x, params):
        super(
            steadystate_solution,
            self).__init__(
            model,
            time,
            x,
            params,
            old=(),
            weights=(
                0.,
            ))


class transient_solution(solution_vector):
    "Transient solution"
    pass


class bdf_solution(transient_solution):
    def __init__(self, model, time, x, params, old, weights):
        super(
            bdf_solution,
            self).__init__(
            model,
            time,
            x,
            params,
            old=old,
            weights=weights)
        assert len(old) + 1 == len(weights)


class bdf1_solution(bdf_solution):
    def __init__(self, model, time, x, params, old, dt):
        super(
            bdf1_solution, self).__init__(
            model, time, x, params, old, weights=[
                1. / dt, -1. / dt])

    def interpolate(self, time):
        ts0 = self.old[0]
        ts1 = self

        assert ts0.time <= time
        assert ts1.time >= time
        k = (time - ts0.time) / (ts1.time - ts0.time)
        k1 = k
        k0 = (1. - k)
        x = ts0.x * k0 + ts1.x * k1
        #xt=ts0.xt() + ts1.xt() * k1
        xt = ts1.xt()
        params = ts1.params
        return const_solution_vector(
            self.model, time=time, x=x, xt=xt, params=params)


class const_solution_vector(solution_vector):
    "Externally provided vectors, which may not solve the model"

    def __init__(self, model, time=0., x=None, xt=None, params=None):
        if x is None:
            x = np.zeros_like(model.X)
        if xt is None:
            xt = np.zeros_like(model.X)
        if params is None:
            params = {}
        super(const_solution_vector, self).__init__(model, time, x, params, ())
        self._xt = xt

    def xt(self):
        return self._xt


class context(object):
    """
    context is top-level object which should be used to run simulations, instead of calling solver and model routines directly.
    context keeps current solution, and optionally, a sequence of obtained solutions (trajectory). It provides easy to use solver, postprocessing, and plotting routines.
    """

    def __init__(self, obj, **kwargs):
        "context can be created as a copy of another context, or by providing model and optionally solution details as keyword arguments"
        self.trajectory = []
        self._output = None
        self.add_hooks = []

        if isinstance(obj, context):
            self.model = obj.model
            self.current = obj.solution
            if kwargs:
                raise ValueError(
                    'keyword arguments not supported when building context from context')
        elif isinstance(obj, model):
            self.model = obj
            self.current = const_solution_vector(obj, **kwargs)
        elif isinstance(obj, solution_vector):
            self.model = obj.model
            self.current = obj
            if kwargs:
                raise ValueError('arguments not supported')
        else:
            raise ValueError('must provide model or context as first argument')
        self._output = None

    # Basic output
    @property
    def x(self):
        return self.current.x

    @property
    def params(self):
        return self.current.params

    @property
    def time(self):
        return self.current.time

    @property
    def xt(self):
        return self.current.xt()

    def _get_solution(self):
        return self.current

    def _set_solution(self, solution):
        self.current = solution
        self._output = None

    solution = property(_get_solution, _set_solution)

    def output(self):
        "Provide output for current solution"
        if self._output is None:
            self._output = self.model.output(
                self.time, self.x, self.xt, self.params, solver=SolverObject())
        return self._output

    # Plotting
    @property
    def pylab(self):
        import oedes.mpl as mpl
        return mpl.forcontext(self)

    def mpl(self, *args, **kwargs):
        import oedes.mpl as mpl
        return mpl.forcontext(self, *args, **kwargs)

    # Interpolating transient simulation
    def _find(self, time):
        i = 0
        if not self.trajectory:
            raise ValueError('not trajectory available')

        def _bsearch(a, cmp):
            i, j, k = 0, len(a), -1
            while i < j:
                k = (i + j) // 2
                c = cmp(a[k])
                if c == 0:
                    break
                elif c < 0:
                    i = k + 1
                else:
                    j = k
            if cmp(a[k]) > 0:
                return k - 1
            return k
        i = _bsearch(self.trajectory, lambda ts: ts.time - time)
        if i + 1 >= len(self.trajectory):
            if self.trajectory[-1].time == time:
                i = len(self.trajectory) - 2
            else:
                raise ValueError('time after last trajectory point')
        return i

    def attime(self, time):
        i = self._find(time)
        return context(self.trajectory[i + 1].interpolate(time))

    # Evaluating functions
    def _eval(self, f):
        if callable(f):
            return f(self)
        elif f == 'time':
            return self.time
        elif f in self.params:
            return self.params[f]
        else:
            output = self.output()
            return self.output()[f]

    def eval(self, *functions):
        return tuple(map(self._eval, functions))

    # Evaluations on trajectory
    def timesteps(self):
        for ts in self.trajectory:
            yield context(ts)

    steps = timesteps

    def teval(self, *functions):
        trajectory = list(self.timesteps())

        def teval_function(f):
            return np.asarray([t._eval(f) for t in trajectory])
        return tuple(map(teval_function, functions))

    def teval_dict(self, *functions):
        values = self.teval(*functions)
        return dict((f, v) for f, v in zip(functions, values))

    # Solving
    def _psolve(self, params, xguess, kwargs):
        kwargs = kwargs.copy()
        kwargs['solver'] = PoissonOnly()
        return solver.solve(self.model, xguess, params, niter=1, **kwargs)

    def _solve(self, xguess, params, pseudo_tmax=1e6, pseudo_dt0=1e-9,
               pseudo_mindt=1e-20, pseudo_maxsteps=200, use_poisson_guess=True, allow_transient=True, **kwargs):
        if use_poisson_guess:
            xguess = self._psolve(params, xguess=xguess, kwargs=kwargs)
        try:
            return solver.solve(self.model, xguess, params, **kwargs)
        except solver.SolverError:
            if not allow_transient:
                raise
            for t, dt, x, xt, out in solver.bdf1adapt_(
                    self.model, xguess, params, 0., pseudo_tmax, pseudo_dt0, mindt=pseudo_mindt, maxsteps=pseudo_maxsteps, **kwargs):
                pass
            return solver.solve(self.model, x, params, **kwargs)

    def add(self):
        self.trajectory.append(self.solution)
        for hook in self.add_hooks:
            hook.add_hook(self)

    def solve(self, params, xguess=None, **kwargs):
        params = dict(params)
        if xguess is None:
            xguess = self.solution.x
        x = self._solve(xguess, params, **kwargs)
        self.solution = steadystate_solution(
            self.model, time=self.solution.time, x=x, params=params)
        self.add()

    def _sweep(self, params, sweeps, xguess, kwargs):
        if not sweeps:
            self.solve(params, xguess=xguess, **kwargs)
            yield ((), self.state())
        else:
            sweep = sweeps[0]
            items = list(sweep(params))
            with_progress = self.progressbar(items, desc=sweep.__name__)
            for x, p in with_progress:
                first = True
                for xi, ci in self._sweep(p, sweeps[1:], xguess, kwargs):
                    if first:
                        xguess = self.x
                    first = False
                    yield (x,) + xi, ci

    def sweeps(self, params, sweeps, xguess=None, **kwargs):
        if xguess is None:
            xguess = self.solution.x
        return self._sweep(params, sweeps, self.x, kwargs)

    def sweep(self, params, sweep, **kwargs):
        for (k,), v in self.sweeps(params, [sweep], **kwargs):
            yield k, v

    def bdf1adapt(self, params, t1, dt, *args, **kwargs):
        params = dict(params)
        for time, dt, x, xt, out in solver.bdf1adapt_(
                self.model, self.solution.x, params, self.solution.time, t1, dt, *args, **kwargs):
            self.solution = bdf1_solution(
                self.model, time, x, params, (self.solution,), dt)
            self.add()
            yield self.state()

    def transientsolve(self, params, timesteps, *args, **kwargs):
        params = dict(params)
        for time, dt, x, xt, out in solver.transientsolve_(
                self.model, self.x, params, timesteps, *args, **kwargs):
            self.solution = bdf1_solution(
                self.model, time, x, params, (self.solution,), dt)
            self.add()
            yield self.state()

    def transient(self, *args, **kwargs):
        for ts in self.progressbar(self.bdf1adapt(
                *args, **kwargs), desc='transient'):
            pass

    def progressbar(self, *args, **kwargs):
        return _progressbar(*args, **kwargs)

    # AC solutions
    def acsolver(self, input_param, output_function_or_name, **kwargs):
        if callable(output_function_or_name):
            output = output_function_or_name
        else:
            assert isinstance(output_function_or_name, str)

            def output(model, time, x, xt, p, full_output):
                return full_output[output_function_or_name]
        return solver.ACSolver(self.model, self.time, self.x,
                               self.xt, self.params, input_param, output, **kwargs)

    # Misc
    def state(self):
        return context(self)


def progressbar(*args, **kwargs):
    leave = kwargs.get('leave', False)
    return progressbar_type(*args, leave=leave, **kwargs)


_progressbar = progressbar


class sweep:
    def __init__(self, parameter_name, values):
        self.parameter_name = parameter_name
        self.__name__ = parameter_name
        self._values = values

    def get_values(self, params):
        if callable(self._values):
            return self._values(params)
        else:
            return self._values

    def __call__(self, params):
        for x in self.get_values(params):
            p = dict(params)
            p[self.parameter_name] = x
            yield (x, p)


def setparam(params, pname, values):
    s = sweep(pname, values)
    return s(params)


def init_notebook():
    global progressbar_type
    progressbar_type = tqdm.tqdm_notebook
