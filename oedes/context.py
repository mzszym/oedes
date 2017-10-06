# -*- coding: utf-8; -*-
#
# oedes - organic electronic device simulator
# Copyright (C) 2017 Marek Zdzislaw Szymanski (marek@marekszymanski.com)
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
from . import util
from .fvm import Poisson
from .models import RestrictedModel

from collections import namedtuple
import numpy as np
import bisect

contextstate = namedtuple('contextstate', ['x', 'xt', 'time', 'params'])


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


_progress_msg = []


class progressinfo:

    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        _progress_msg.append(self.msg)

    def __exit__(self, type, value, traceback):
        _progress_msg.pop()


class context:

    def __init__(self, model_or_context, **kwargs):
        if isinstance(model_or_context, context):
            ctx = model_or_context
            self.model = ctx.model
            self.x = ctx.x
            self.xt = ctx.xt
            self.time = ctx.time
            self.params = ctx.params
            self._output = ctx._output
            self.trajectory = ctx.trajectory
            self._tsdict = dict(ctx._tsdict)
        else:
            model = model_or_context
            self.model = model
            self.x = np.zeros_like(model.X)
            self.xt = np.zeros_like(model.X)
            self.time = 0.
            self.params = {}
            self._output = None
            self.trajectory = []
            self._tsdict = {}
        self.set(**kwargs)

    def progress(self, ts):
        util.progress('%s time=%s' % (' '.join(_progress_msg), ts.time))

    def state(self):
        return context(self.model, x=self.x, xt=self.xt,
                       time=self.time, params=self.params)

    def restore(self, state):
        assert self.model is state.model
        self.set(time=state.time, x=state.x, xt=state.xt, params=state.params)

    def set(self, time=None, x=None, xt=None, params=None):
        if time is not None:
            self.time = time
        if x is not None:
            self.x = x
        if xt is not None:
            self.xt = xt
        if params is not None:
            self.params = params
        self._output = None
        self._acsolver = None

    def bdf1adapt(self, params, t1, dt, *args, **kwargs):
        params = dict(params)
        self.trajectory.append(contextstate(
            time=self.time, x=self.x, xt=self.xt, params=params))
        for time, x, xt, out in solver.bdf1adapt(
                self.model, self.x, params, self.time, t1, dt, *args, **kwargs):
            self.set(time=time, x=x, xt=xt, params=params)
            self.trajectory.append(self.state())
            yield self.state()

    def transientsolve(self, params, timesteps, *args, **kwargs):
        params = dict(params)
        self.trajectory.append(contextstate(
            time=self.time, x=self.x, xt=self.xt, params=params))
        for time, x, xt, out in solver.transientsolve(
                self.model, self.x, params, timesteps, *args, **kwargs):
            self.set(time=time, x=x, xt=xt, params=params)
            self.trajectory.append(self.state())
            yield self.state()

    def transient(self, *args, **kwargs):
        for ts in self.bdf1adapt(*args, **kwargs):
            self.progress(ts)

    def _psolve(self, params, xguess):
        eqs = [
            name for name,
            eq in self.model.findeqs(
                lambda eq:isinstance(
                    eq,
                    Poisson),
                return_names=True)]
        if not eqs:
            return xguess
        r = RestrictedModel(self.model, eqs, xguess, 0. * xguess)
        phi = solver.solve(r, r.X, params)
        return r.x_to_model(phi)

    def _solve(self, xguess, params, pseudo_tmax=1e6, pseudo_dt0=1e-9,
               pseudo_mindt=1e-15, pseudo_maxsteps=200, use_poisson_guess=True, **kwargs):
        if use_poisson_guess:
            xguess = self._psolve(params, xguess=xguess)
        try:
            return solver.solve(self.model, xguess, params, **kwargs)
        except solver.SolverError:
            for t, x, xt, out in solver.bdf1adapt(
                    self.model, xguess, params, 0., pseudo_tmax, pseudo_dt0, mindt=pseudo_mindt, maxsteps=pseudo_maxsteps, **kwargs):
                pass
            return solver.solve(self.model, x, params, **kwargs)

    def sweep(self, parameters, **kwargs):
        xguess = self.x
        for params in parameters:
            x = self._solve(xguess, params, **kwargs)
            xguess = x
            self.set(time=0., x=x, params=params)
            self.trajectory.append(self.state())
            yield self.state()

    def solve(self, params, xguess=None, **kwargs):
        params = dict(params)
        if xguess is None:
            xguess = self.x
        x = self._solve(xguess, params, **kwargs)
        self.set(time=0., x=x, xt=np.zeros_like(x), params=params)
        self.trajectory.append(self.state())

    def output(self):
        if self._output is None:
            self._output = self.model.output(
                self.time, self.x, self.xt, self.params)
        return self._output

    @property
    def pylab(self):
        import oedes.mpl as mpl
        return mpl.forcontext(self)

    def mpl(self, fig, ax):
        import oedes.mpl as mpl
        return mpl.forcontext(self, fig, ax)

    def attime(self, time):
        i = 0
        if not self.trajectory:
            raise ValueError
        i = _bsearch(self.trajectory, lambda ts: ts.time - time)
        if i < 0:
            raise ValueError
        ts0 = self.trajectory[i]
        assert ts0.time <= time
        if ts0.time < time:
            if (i + 1) >= len(self.trajectory):
                raise ValueError()
            ts1 = self.trajectory[i + 1]
            assert ts1.time > time
            assert ts0.params is ts1.params
            k = (time - ts0.time) / (ts1.time - ts0.time)
            ts = contextstate(time=time, x=ts0.x + (ts1.x - ts0.x) * k,
                              xt=ts0.xt + k * (ts1.xt - ts0.xt), params=ts1.params)
        else:
            ts = ts0
        return context(self.model, time=ts.time, x=ts.x,
                       xt=ts.xt, params=ts.params)

    def timesteps(self):
        for ts in self.trajectory:
            if id(ts) not in self._tsdict:
                self._tsdict[id(ts)] = context(
                    self.model, time=ts.time, x=ts.x, xt=ts.xt, params=ts.params)
            yield self._tsdict[id(ts)]

    def eval_function(self, f):
        if callable(f):
            return f(self)
        elif f in self.output():
            return self.output()[f]
        elif f in self.params:
            return self.params[f]
        elif f == 'time':
            return self.time
        else:
            raise RuntimeError('cannot find %s' % f)

    def teval_function(self, f):
        return np.asarray([ts.eval_function(f) for ts in self.timesteps()])

    def eval(self, *functions):
        return [self.eval_function(f) for f in functions]

    def teval(self, *functions):
        return [self.teval_function(f) for f in functions]

    def acsolver(self, input_param, output_function_or_name, **kwargs):
        if callable(output_function_or_name):
            output = output_function_or_name
        else:
            assert isinstance(output_function_or_name, str)

            def output(model, time, x, xt, p, full_output):
                return full_output[output_function_or_name]
        return solver.ACSolver(self.model, self.time, self.x,
                               self.xt, self.params, input_param, output, **kwargs)


def setparam(params, pname, values):
    for x in values:
        p = dict(params)
        p[pname] = x
        yield p
