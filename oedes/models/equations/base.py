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

from oedes.fvm import FVMConservationEquation, FVMBoundaryEquation, DummyConvergenceTest
from oedes.utils import Funcall, Computation

__all__ = [
    'DelegateConvergenceTest',
    'Equation',
    'ConservationEquation',
    'BoundaryEquation',
    'Calculation',
    'Funcall']


class Equation(Computation):
    def __init__(self, name=None):
        self.name = name
        self.convergenceTest = None

    @property
    def prefix(self):
        return self.name

    def all_equations(self, args):
        yield (args.push(prefix=self.name), self)


class DelegateConvergenceTest(object):
    def __init__(self, target):
        self.target = target

    def testEquationNew(self, eq, F, x, dx, report):
        return self.target.convergenceTest.testEquationNew(
            eq, F, x, dx, report)

    def testBoundaryNew(self, eq, bc, F, x, dx, report):
        return self.target.convergenceTest.testBoundaryNew(
            eq, bc, F, x, dx, report)


class ConservationEquation(Equation):
    def __init__(self, mesh, name=None):
        super(ConservationEquation, self).__init__(name)
        self.defaultBCConvergenceTest = None
        self.mesh = mesh
        self.bc = []

    def load(self, ctx, eq):
        vars = ctx.varsOf(eq)
        vars['x'] = ctx.gvars['x'][eq.idx]
        vars['xt'] = ctx.gvars['xt'][eq.idx]

    def evaluate_bc(self, ctx, eq, v, fluxes, FdS_boundary,
                    celltransient_boundary=0., cellsource_boundary=0.):
        conservation = -FdS_boundary + eq.mesh.cells['volume'][
            eq.mesh.boundary.idx] * (celltransient_boundary - cellsource_boundary)
        ctx.gvars['bc_conservation'].append((eq.boundary_labels, conservation))


class BoundaryEquation(Equation):
    def __init__(self, name):
        super(BoundaryEquation, self).__init__(name)
        self._owner_eq_weak = None
        self._owner_eq = None

    @property
    def owner_eq(self):
        if self._owner_eq is not None:
            assert self._owner_eq_weak is None, 'cannot assign both references to owner'
            return self._owner_eq
        assert self._owner_eq_weak is not None, 'no reference to owner'
        return self._owner_eq_weak()

    def build(self, builder):
        obj = builder.newDirichletBC(builder.get(self.owner_eq), self.name)
        self._build_bc(builder, obj)
        return obj

    def evaluate(self, ctx, eq):
        raise NotImplementedError()

    def _init(self, builder, obj):
        obj.owner_eq = builder.get(self.owner_eq)
        obj.evaluate.depends(obj.owner_eq.load)

    def _build_bc(self, builder, obj):
        obj.convergenceTest = DelegateConvergenceTest(self)
        obj.evaluate = Funcall(self.evaluate, obj)
        builder.add(self, obj)
        builder.addInitializer(self._init, obj)
        builder.addEvaluation(obj.evaluate)


class Calculation(Equation):
    def build(self, builder):
        obj = builder.newGeneralDiscreteEquation(0)
        obj.convergenceTest = DummyConvergenceTest()
        builder.add(self, obj)
        return obj
