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

from oedes.ad import getitem, sparsesum, dot
from oedes.utils import EquationWithMesh, Equation
import itertools
import weakref

__all__ = ['ConservationEquation', 'BoundaryEquation']


class ConservationEquation(EquationWithMesh):
    def __init__(self, mesh=None, name=None):
        super(ConservationEquation, self).__init__(mesh, name)
        self.defaultBCConvergenceTest = None
        self.bc = []

    def load(self, ctx, eq):
        super(ConservationEquation, self).load(ctx, eq)
        vars = ctx.varsOf(eq)
        vars['boundary_sources'] = []
        vars['boundary_FdS'] = []

    def _evaluate_bc(self, ctx, eq, FdS_boundary,
                     celltransient_boundary=0., cellsource_boundary=0.):
        conservation = -FdS_boundary + eq.mesh.cells['volume'][
            eq.mesh.boundary.idx] * (celltransient_boundary - cellsource_boundary)
        ctx.gvars['bc_conservation'].append((eq.boundary_labels, conservation))

    def residuals(self, ctx, eq, flux, source=None, transient=0.):
        variables = ctx.varsOf(eq)
        yield eq.residuals(eq.mesh.internal, flux, cellsource=source, celltransient=transient)
        n = len(eq.mesh.boundary.cells)
        bc_FdS = sparsesum(n, variables['boundary_FdS'])
        bc_source = sparsesum(n, variables['boundary_sources'])
        if flux is not None:
            bc_FdS = dot(eq.mesh.boundary.fluxsum, flux) + bc_FdS
        if source is not None:
            bc_source = getitem(source, eq.mesh.boundary.idx) + bc_source
        bc_transient = getitem(transient, eq.mesh.boundary.idx)
        variables['total_boundary_FdS'] = bc_FdS
        variables['total_boundary_sources'] = bc_source
        variables['total_boundary_transient'] = bc_transient
        self._evaluate_bc(
            ctx,
            eq,
            bc_FdS,
            cellsource_boundary=bc_source,
            celltransient_boundary=bc_transient)

    def identity(self, ctx, eq):
        yield eq.identity(ctx.varsOf(eq)['x'])

    def all_equations(self, args):
        def bcs():
            for bc in self.bc:
                yield bc
        for eq in super(ConservationEquation, self).all_equations(args):
            yield eq
        for bc in self.bc:
            assert bc._owner_eq is None
            if bc._owner_eq_weak is None:
                bc._owner_eq_weak = weakref.ref(self)
            else:
                assert bc._owner_eq_weak() is self
            for eq in bc.all_equations(args):
                yield eq


class BoundaryEquation(Equation):
    def __init__(self, name, owner=None):
        super(BoundaryEquation, self).__init__(name)
        self._owner_eq_weak = None
        self._owner_eq = owner

    @property
    def owner_eq(self):
        if self._owner_eq is not None:
            assert self._owner_eq_weak is None, 'cannot assign both references to owner'
            return self._owner_eq
        assert self._owner_eq_weak is not None, 'no reference to owner'
        return self._owner_eq_weak()

    def newDiscreteEq(self, builder):
        return builder.newDirichletBC(builder.get(self.owner_eq), self.name)

    def evaluate(self, ctx, eq):
        raise NotImplementedError()

    def initDiscreteEq(self, builder, obj):
        super(BoundaryEquation, self).initDiscreteEq(builder, obj)
        obj.convergenceTest = builder.newDelegateConvergenceTest(self.owner_eq)
        obj.owner_eq = builder.get(self.owner_eq)
        obj.evaluate.depends(obj.owner_eq.load)
        obj.owner_eq.alldone.depends(obj.evaluate)
