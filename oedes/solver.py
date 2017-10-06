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

import numpy as np
from .ad import *
from .param import paramvector
import scipy.sparse.linalg
import scipy.sparse
import warnings
import logging
from collections import defaultdict
from .util import TODOWarning


class SolverError(RuntimeError):

    def __init__(self, *args):
        RuntimeError.__init__(self, *args)


logger = logging.getLogger('oedes.solver')

# In[41]:

# <api>


def _convert(A, b):
    if A.dtype == np.longdouble:
        A = scipy.sparse.csr_matrix(A, dtype=np.double)
    elif A.dtype == np.longcomplex:
        A = scipy.sparse.csr_matrix(A, dtype=np.complex)
    if b.dtype == np.longdouble:
        b = np.asarray(b, dtype=np.double)
    elif b.dtype == np.longcomplex:
        b = np.asarray(b, dtype=np.complex)
    return A, b


def spsolve_scipy(A, b):
    A, b = _convert(A, b)
    with warnings.catch_warnings():
        warnings.simplefilter(
            "error", category=scipy.sparse.linalg.MatrixRankWarning)
        try:
            return scipy.sparse.linalg.spsolve(A, b, use_umfpack=False)
        except BaseException:
            raise SolverError('scipy.sparse.linalg.spsolve failed')


def _explain(report):
    columns = ['atol', 'rtol', 'maxabs', 'maxrel', 'maxabs_nc', 'maxrel_nc']
    header = '%-26s C?  atol    rtol   maxabs  maxrel nc_maxabs nc_maxrel' % 'equation'

    def makerow(k):
        d = report[k]
        if len(k) > 26:
            k = '>' + k[-25:]
        items = [k, ] + [d[u] for u in columns]
        numbers = tuple('{:^7}'.format('%1.1e' % d[u]) for u in columns)
        return '%-26s %.1s %s' % (k, d['converged'], ' '.join(numbers))
    return '\n'.join([header] + [makerow(k) for k in sorted(report.keys())])


def _matrixsolve(spsolve, A, b, scaling):
    xscaling, fscaling = scaling
    Mr = scipy.sparse.diags(
        [xscaling],
        offsets=[0],
        format='csr',
        dtype=xscaling.dtype)
    Ml = scipy.sparse.diags(
        [fscaling],
        offsets=[0],
        format='csr',
        dtype=fscaling.dtype)
    x_ = spsolve(Ml.dot(A.dot(Mr)), Ml.dot(b))
    return Mr.dot(x_)


def solve(model, x0, params, tconst=0., tshift=0., time=0.,
          maxiter=10, spsolve=spsolve_scipy, solver=None):
    """This solves system F(x)=0 with x0 as initial guess

    Transient solution is possible by assuming xt=x*tshift+tconst.
    Both tshift and tconst should be zero for stationary.

    If should does not specify :convergence_test: (F,x,dx,params)

    Returns x,xt.
    """

    F = model.residuals
    xscaling, fscaling = model.scaling(params)
    convergence_test = model.converged
    update = model.update

    logger = logging.getLogger('oedes.solver.nonlinear')

    # _ denotes scaled variables
    x_ = x0 / xscaling
    dx_ = None
    itno = 0
    report = None
    while True:
        itno = itno + 1
        if not np.alltrue(np.isfinite(x_)):
            raise SolverError('solution diverged')
        if itno > maxiter:
            logger.info(
                'did not converge in %d iterations\n%s' %
                (maxiter, _explain(report)))
            raise SolverError('did not converge in %d iterations' % maxiter)
        adx = forward.seed_sparse_gradient(x_) * xscaling
        f_unscaled = F(time, adx, adx * tshift + tconst, params, solver=solver)
        f = f_unscaled * fscaling
        logger.debug("iteration %d |F|=%e" %
                     (itno, np.linalg.norm(f_unscaled.value)))
        if dx_ is not None:
            report = dict()
            if convergence_test(f_unscaled.value, adx.value,
                                dx_ * xscaling, params, report=report):
                break
        dx_ = spsolve(f.gradient.tocsr(), -f.value)
        if not np.alltrue(np.isfinite(dx_)):
            msg = 'linear solver failed (NaN)'
            logger.info(msg)
            raise SolverError(msg)
        # Update
        x_ += dx_
        x = x_ * xscaling
        update(x, params)
        x_ = x / xscaling
    x = x_ * xscaling
    logger.info('converged in %d iterations' % itno)
    return x


# In[42]:

class TransientSolver:
    dt = None

# <api>


def bdf1adapt(model, x, params, t, t1, dt, mindt=1e-15, xt=None, dtrlim=(0.5, 2.), relfail=10., reltol=1e-2, abstol=1e-15, dtrfailsolve=0.1,
              dtrfaillte=0.1, weight=None, skip=2, use_predictor=False, final_matchstep=True, solve=solve, maxsteps=np.inf, maxdt=np.inf, **kwargs):
    """
    Adaptive transient solver based on BDF1. Solve from time :t0: to at least :t1:, with initial timestep :dt:.
    :xt: is xt at t, can be None.
    :mindt: is minimum dt, below which calculation will be aborted.

    The goal is to keep norm of solution LTE (local truncation error) satisfying:
    norm(lte) <= reltol*norm(solution)+abstol

    Variables are taken into account scaled by :weight:.

    If norm(lte) > reject*(reltol*norm(solution)+abstol), the timestep is rejected and new dt tried faillte.

    The timestep in after successful step is in range (dtrlim*dt).

    Takes number of initial points to :skip.

    This method is inspired by Trilinos Rythmos::FirstOrderErrorStepControlStrategy.
    """
    if weight is None:
        weight = model.transientvar
    failures = 0
    solver = TransientSolver()
    step = 0
    logger = logging.getLogger('oedes.solver.bdf1dapt')
    while t < t1:
        # t,x is current solution
        if dt < mindt:
            raise SolverError(
                'transient timestep smaller than minimum timestep %s' % mindt)
        if step >= maxsteps:
            raise SolverError(
                'maximum number of %s timesteps reached' % maxsteps)
        assert dt <= maxdt
        tn = t + dt
        if final_matchstep and tn > t1:
            tn = t1
            dt = t1 - t
        # Calculate predicted solution by forward Euler
        xnp = None
        if skip:
            skip -= 1
        elif xt is not None:
            xnp = x + xt * dt
        # Calculate new solution by backward Euler xt=(xn-x)/dt
        try:
            if use_predictor:
                initial_guess = xnp
            else:
                initial_guess = x
            solver.dt = dt
            xn = solve(model, initial_guess, params, tconst=-x / dt,
                       tshift=1. / dt, time=tn, solver=solver, **kwargs)
            xtn = (xn - x) / dt
        except SolverError:
            dt *= dtrfailsolve
            logger.info('failure in solve, reducing timestep to %e' % (dt))
            failures += 1
            continue
        if xnp is not None:
            dn = (xn - xnp) / 2
            goalnorm = reltol * np.linalg.norm(xn * weight) + abstol
            dnorm = np.linalg.norm(dn * weight)
            if dnorm > relfail * goalnorm:
                dt *= dtrfaillte
                failures += 1
                logger.info('error too big, reducing timestep to %e' % (dt))
                continue
            e = goalnorm / dnorm
            dt = np.clip(dt * np.sqrt(e), min(mindt,
                                              dtrlim[0] * dt), min(dtrlim[1] * dt, maxdt))
        t, x, xt = tn, xn, xtn

        def output():
            d = model.output(t, x, xt, params)
            d['tsadapt.failures'] = failures
            d['tsadapt.dt'] = dt
            d['tsadapt.time'] = t
            return d
        yield (t, x, xt, output)


# In[43]:

# <api>
def transientsolve(model, x0, params, timesteps,
                   solve=solve, mindt=1e-15, force=False, **kwargs):
    "Return sequence of transient solutions (t,x(t),xt,output) for predefined timesteps"
    x = x0
    solver = TransientSolver()
    logger = logging.getLogger('oedes.solver.transientsolve')
    for i, dt in enumerate(np.diff(timesteps)):
        t0, t1 = timesteps[i:i + 2]
        try:
            logger.info('t=%e dt=%e' % (t0, dt))
            solver.dt = dt
            xn = solve(model, x, params, tconst=-x / dt,
                       tshift=1. / dt, time=t1, solver=solver, **kwargs)
            x, xt = xn, (xn - x) / dt
            r = (timesteps[i + 1], x, xt,
                 lambda: model.output(t1, x, xt, params))
        except SolverError:
            if not force:
                raise
            logger.info('timestep failed')
            if 0.5 * (t1 - t0) < mindt:
                raise
            # try with half timestep
            _, r = list(transientsolve(model, x, params, [
                        t0, 0.5 * (t0 + t1), t1], **kwargs))
            x, xt = r[1:3]
        yield r


def asarrays(generator, outputs=['J'], monitor=lambda t, x, xt, out: None):
    data = dict([(o, []) for o in outputs])
    assert 'time' not in data
    data['time'] = []
    for t, x, xt, outf in generator:
        out = outf()
        data['time'].append(t)
        for k in outputs:
            data[k].append(out[k])
        monitor(t, x, xt, out)
    return dict([(k, np.asarray(data[k])) for k in data])


def interpolatearrays(data, t):
    time = data['time']
    i = np.searchsorted(time, t, side='left') - 1
    t0, t1 = time[i:i + 2]
    a = (t - t0) / (t1 - t0)
    assert a >= 0. and a <= 1.
    return dict((k, (data[k][i + 1] - data[k][i]) * a + data[k][i])
                for k in data.keys())


class ACSolver:
    def __init__(self, model, time, x, xt, params,
                 inparam, outfunc, spsolve=spsolve_scipy):
        n = len(x)
        sparams = paramvector([inparam])
        p = sparams.values(params)
        u = np.concatenate([x, xt, p])
        adu = forward.seed_sparse_gradient(u)
        x, xt = adu[:n], adu[n:2 * n]
        p = sparams.asdict(params, adu[2 * n:])
        full_output = {}
        F = model.residuals(
            time,
            x,
            xt,
            p,
            full_output=full_output,
            solver=self)
        output = outfunc
        J = output(model, time, x, xt, p, full_output)

        def split(a):
            A = a.gradient.tocsr()
            return a.value, (A[:, :n], A[:, n:2 * n], A[:, 2 * n:])
        _, (self.F_x, self.F_xt, F_p) = split(F)
        self.J, (self.J_x, self.J_xt, self.J_p) = split(J)
        self.rhs = -F_p.todense()
        self.spsolve = spsolve
        self.scaling = model.scaling(params)

    def solve(self, omega):
        iw = 1.j * omega
        A = self.F_x + iw * self.F_xt
        X = _matrixsolve(self.spsolve, A, self.rhs, self.scaling)
        J = (self.J_x + 1.j * omega * self.J_xt).dot(X) + self.J_p
        assert J.shape == (1, 1)
        return X, J[0, 0]


def acsolve(model, time, x, xt, params, inparam, outfunc, omegas, **kwargs):
    s = ACSolver(model, time, x, xt, params, inparam, outfunc, **kwargs)
    for w in omegas:
        _, j = s.solve(w)
        yield (w, j)
