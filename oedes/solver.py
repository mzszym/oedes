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

import numpy as np
from .ad import *
from .param import paramvector
from sparsegrad.impl import scipy
if hasattr(scipy.sparse, 'linalg'):
    linalg = scipy.sparse.linalg
else:
    from scipy.sparse import linalg
import numpy as np
import warnings
import logging
from collections import defaultdict
from . import logs


class SolverError(RuntimeError):

    def __init__(self, *args):
        RuntimeError.__init__(self, *args)


class SolverObject(object):
    def __init__(self):
        pass

    @property
    def poissonOnly(self):
        return False


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


def spsolve_scipy_(A, b):
    A, b = _convert(A, b)
    with warnings.catch_warnings():
        warnings.simplefilter(
            "error", category=linalg.MatrixRankWarning)
        logs.linear.info('Solving linear equation using scipy sparse solver')
        logs.linear.info(
            'System matrix has shape {shape} with {nnz} nonzero entries of dtype {dtype}'.format(
                shape=A.shape, nnz=A.nnz, dtype=A.dtype))
        try:
            return linalg.spsolve(A, b, use_umfpack=False)
        except BaseException:
            raise SolverError('spsolve failed')


def spsolve_scipy(A, b):
    return spsolve_scipy_(A, b)


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
          maxiter=10, spsolve=spsolve_scipy, solver=None, niter=None):
    """This solves system F(x)=0 with x0 as initial guess

    Transient solution is possible by assuming xt=x*tshift+tconst.
    Both tshift and tconst should be zero for stationary.

    If should does not specify :convergence_test: (F,x,dx,params)

    Returns x,xt.
    """

    if solver is None:
        solver = SolverObject()

    F = model.residuals
    xscaling, fscaling = model.scaling(params)
    convergence_test = model.converged
    update = model.update

    logger = logs.nonlinear

    logs.timestepping.info('Solving %s' % solver)

    # _ denotes scaled variables
    x_ = x0 / xscaling
    dx_ = None
    itno = 0
    report = None
    timing_logger = logs.nonlinear.getChild('timing')
    while True:
        itno = itno + 1
        if not np.alltrue(np.isfinite(x_)):
            raise SolverError('solution diverged')
        if itno > maxiter:
            nclist = [k for k, v in report.items() if not v['converged']]
            logger.info(
                'Did not converge in %d iterations: %s' %
                (maxiter, nclist))
            raise SolverError('did not converge in %d iterations' % maxiter)
        timer = logs.Timer()
        adx = forward.seed_sparse_gradient(x_) * xscaling
        f_unscaled = F(time, adx, adx * tshift + tconst, params, solver=solver)
        f = f_unscaled * fscaling
        logger.info("Nonlinear iteration %d, residual norm |F|=%e (unscaled %e)" %
                    (itno, np.linalg.norm(f.value), np.linalg.norm(f_unscaled.value)))
        if dx_ is not None:
            report = dict()
            if convergence_test(f_unscaled.value, adx.value,
                                dx_ * xscaling, params, report=report):
                break
        timing_logger.info(
            'Calculation of Jacobian matrix and F took %s' %
            timer)
        timer = logs.Timer()
        dx_ = spsolve(f.gradient.tocsr(), -f.value)
        timing_logger.info('Linear solve took %s' % timer)
        if not np.alltrue(np.isfinite(dx_)):
            msg = 'linear solver failed (NaN)'
            logger.info(msg)
            raise SolverError(msg)
        x_ = x_ + dx_
        x = x_ * xscaling
        update(x, params)
        x_ = x / xscaling
        if niter is not None and itno >= niter:
            break
    x = x_ * xscaling
    #logger.info('converged in %d iterations' % itno)
    return x


def bdf1adapt_(model, x, params, t, t1, dt, mindt=1e-15, xt=None, dtrlim=(0.5, 2.), relfail=10., reltol=1e-2, abstol=1e-15, dtrfailsolve=0.1,
               dtrfaillte=0.1, weight=None, skip=2, use_predictor=False, final_matchstep=True, solve=solve, maxsteps=1000, maxdt=np.inf, solver=None, **kwargs):
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
    step = 1
    attempt = 1
    logger = logs.timestepping
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
        logger.info(
            'Timestep %d (attempt %d) at time %e with dt %e' %
            (step, attempt, t, dt))
        attempt += 1
        try:
            if use_predictor:
                initial_guess = xnp
            else:
                initial_guess = x
            xn = solve(model, initial_guess, params, tconst=-x / dt,
                       tshift=1. / dt, time=tn, solver=solver, **kwargs)
            xtn = (xn - x) / dt
            dt_ = dt
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
            e = goalnorm / (dnorm + 1e-100)
            dt = np.clip(dt * np.sqrt(e), min(mindt,
                                              dtrlim[0] * dt), min(dtrlim[1] * dt, maxdt))
        t, x, xt = tn, xn, xtn
        step += 1

        def output():
            d = model.output(t, x, xt, params)
            d['tsadapt.failures'] = failures
            d['tsadapt.dt'] = dt
            d['tsadapt.time'] = t
            return d
        yield (t, dt_, x, xt, output)


def bdf1adapt(*args, **kwargs):
    for t, dt, x, xt, output in bdf1adapt_(*args, **kwargs):
        yield t, x, xt, output

# In[43]:

# <api>


def transientsolve_(model, x0, params, timesteps,
                    solve=solve, mindt=1e-15, force=False, maxsteps=100, solver=None, **kwargs):
    "Return sequence of transient solutions (t,dt,x(t),xt,output) for predefined timesteps"
    x = x0
    logger = logs.timestepping
    assert len(timesteps) - 1 <= maxsteps, 'not enough steps allowed'

    def check_step(step):
        if step > maxsteps:
            raise SolverError(
                'maximum number of %s timesteps reached' % maxsteps)
    step = 0
    for i, dt in enumerate(np.diff(timesteps)):
        check_step(step)
        t0, t1 = timesteps[i:i + 2]
        try:
            logger.info('t=%e dt=%e' % (t0, dt))
            xn = solve(model, x, params, tconst=-x / dt,
                       tshift=1. / dt, time=t1, solver=solver, **kwargs)
            x, xt = xn, (xn - x) / dt
            step += 1
            yield (timesteps[i + 1], dt, x, xt,
                   lambda: model.output(t1, x, xt, params))
        except SolverError:
            if not force:
                raise
            logger.info('timestep failed')
            if 0.5 * (t1 - t0) < mindt:
                raise
            for t, dt, x, xt, output in transientsolve_(
                    model, x, params, [t0, 0.5 * (t0 + t1), t1], maxsteps=np.inf, **kwargs):
                check_step(step)
                step += 1
                yield t, dt, x, xt, output


def transientsolve(*args, **kwargs):
    for t, dt, x, xt, output in transientsolve_(*args, **kwargs):
        yield t, x, xt, output


class ACSolver:
    def __init__(self, model, time, x, xt, params,
                 inparam, outfunc, spsolve=spsolve_scipy, solver=None):
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
            solver=solver)
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
