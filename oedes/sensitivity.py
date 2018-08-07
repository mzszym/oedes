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

from .context import context, const_solution_vector
from . import ad
from .solver import _matrixsolve, spsolve_scipy, SolverObject
import numpy as np

__all__ = ['sensitivity_analysis', 'add_sensitivity']


class sensitivity_analysis(SolverObject):
    def __init__(self, sparams, function='J', spsolve=spsolve_scipy):
        self.sensitivity = {}
        self.function = function
        self.sparams = sparams
        self.spsolve = spsolve
        self._g = {}
        self._dg = {}

    def goal(self, solution, time, x, xt, params, full_output):
        return full_output[self.function]

    def process(self, solution):
        n = len(solution.x)
        nparams = self.sparams.values(solution.params)

        def gets(obj):
            if isinstance(obj, const_solution_vector):
                return np.asarray(0.)
            else:
                return self.sensitivity[id(obj)]
        if solution.old:
            nold = sum(o.x * w for o,
                       w in zip(solution.old,
                                solution.weights[1:]))
            dxold = sum(gets(o) *
                        w for o, w in zip(solution.old, solution.weights[1:]))
        else:
            nold = np.zeros_like(solution.model.X)
            dxold = np.asarray(0.)
        if not dxold.shape:
            dxold = dxold * np.ones((n, len(nparams)))
        nvec = np.concatenate([solution.x, nold, nparams])
        vec = ad.forward.seed_sparse_gradient(nvec)
        x = vec[:n]
        xold = vec[n:2 * n]
        xt = solution.weights[0] * x + xold
        p = vec[2 * n:]
        params = self.sparams.asdict(solution.params, p)
        full_output = {}
        F = solution.model.residuals(
            solution.time, x, xt, params, full_output=full_output, solver=self)
        G = self.goal(solution, solution.time, x, xt, params, full_output)

        def split(g):
            g = g.gradient.tocsr()
            return g[:, :n], g[:, n:2 * n], g[:, 2 * n:]
        dF_dx, dF_dxold, dF_dp = split(F)
        dG_dx, dG_dxold, dG_dp = split(G)
        b = -dF_dp - dF_dxold.dot(dxold)
        scaling = solution.model.scaling(solution.params)
        dx_dp = _matrixsolve(self.spsolve, dF_dx, b, scaling)
        self.sensitivity[id(solution)] = dx_dp
        d = dG_dx.dot(dx_dp) + dG_dxold.dot(dxold) + dG_dp
        self._g[id(solution)] = G.value
        self._dg[id(solution)] = d

    def add_hook(self, context):
        return self.process(context.solution)

    def g(self, c):
        return [self._g[id(u)] for u in c.trajectory]

    def dg(self, c):
        return [self._dg[id(u)] for u in c.trajectory]


def add_sensitivity(context, *args, **kwargs):
    sa = sensitivity_analysis(*args, **kwargs)
    context.add_hooks.append(sa)
    return sa
