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
from .cell import FVMBoundaryEquation, FVMConservationEquation
import scipy.sparse.csgraph
from oedes.ad import sparsesum
from oedes.model import model


class FVMEvalContext(object):
    def __init__(self, evaluator, target):
        self.target = target
        self.evaluator = evaluator
        self.bc_conservation = []

    def finalize(self):
        return self.evaluator.finalize(self.target, self.bc_conservation)


class KillVars(object):
    def __init__(self, obj):
        self.obj = obj

    def __call__(self, ctx):
        ctx.killVars(self.obj)


class FVMEvaluator(model):
    def __init__(self, builder, context_type):
        self.equations = builder.equations
        self.ndof = builder.ndof

        self._init_boundary()

        X = np.zeros(builder.ndof)
        self.X = X

        transientvar = X.copy()
        for eq in self.equations:
            transientvar[eq.idx] = eq.transientvar
        self.transientvar = transientvar
        self.plan = tuple(builder.fungraph.plan(kill_func=KillVars))
        self.context_type = context_type

    def _all_bc(self):
        for eq in self.equations:
            if isinstance(eq, FVMBoundaryEquation):
                yield eq

    def _all_conservation(self):
        for eq in self.equations:
            if isinstance(eq, FVMConservationEquation):
                yield eq

    def _init_boundary(self):
        i = [np.zeros(0, dtype=np.int)]
        j = [np.zeros(0, dtype=np.int)]
        for bc in self._all_bc():
            valid = bc.conservation_to_dof >= 0
            i.append(bc._getDof()[valid])
            j.append(bc.conservation_to_dof[valid])
            #eq.bc_free_dof = np.arange(len(eq.mesh.boundary.idx))[fdof]
        i = np.hstack(i)
        j = np.hstack(j)
        # optimize: should only create vector for boundary items
        g = scipy.sparse.csr_matrix(
            (np.ones_like(i), (i, j)), shape=(self.ndof,) * 2)
        nlabels, labels = scipy.sparse.csgraph.connected_components(
            g, directed=False, return_labels=True)
        self.bc_label_volume = sparsesum(nlabels, ((labels[eq.idx], eq.mesh.cells[
                                         'volume']) for eq in self.equations if isinstance(eq, FVMConservationEquation)))
        self.bc_labels = labels
        for eq in self._all_conservation():
            eq.boundary_labels = self.bc_labels[eq.idx[eq.mesh.boundary.idx]]

    def createContext(self, target):
        return FVMEvalContext(self, target)

    def finalize(self, target, bc_conservation):
        bc_label_conservation = sparsesum(
            len(self.bc_labels), bc_conservation)
        for eq in self.equations:
            if isinstance(eq, FVMConservationEquation):
                bc_free_dof = np.arange(len(eq.mesh.boundary.idx))[
                    eq.bc_dof_is_free]
                i = eq.idx[eq.mesh.boundary.idx[bc_free_dof]]
                j = self.bc_labels[i]
                yield i, bc_label_conservation[j] * (1. / self.bc_label_volume[j])

    def scaling(self, params):
        xscaling = np.ones_like(self.X)
        fscaling = np.ones_like(self.X)
        for eq in self.equations:
            eq.scaling(xscaling, fscaling)
        return xscaling, fscaling

    def update(self, x, params):
        for eq in self.equations:
            eq.update(x, params)

    def evaluate(self, time, x, xt, params, full_output=None, solver=None):
        ectx = self.createContext(None)
        ctx = self.context_type(
            params,
            full_output,
            gvars=dict(
                time=time,
                x=x,
                xt=xt,
                params=params,
                full_output=full_output,
                solver=solver,
                bc_conservation=ectx.bc_conservation))

        for p in self.plan:
            itr = p(ctx)
            if itr is not None:
                for ix, v in itr:
                    yield ix, v
        for ix, v in ectx.finalize():
            yield ix, v
        # for ix,v in self.generate_bc_conservation(ctx):
        #    yield ix,v

    def converged(self, residuals, x, dx, params, report, equations=None):
        if equations is None:
            equations = self.equations
        c = all([eq.testConverged(residuals, x, dx, report)
                 for eq in equations])
        return c
