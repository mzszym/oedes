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
from .base import BoundaryEquation
from .poisson import PoissonEquation
from oedes.utils import Funcall
from oedes import functions
from oedes.ad import where, getitem, sum
from oedes.models import solver

__all__ = [
    'DirichletBC',
    'DirichletBCWithFunction',
    'DirichletFromParams',
    'Zero',
    'DirichletFromParams',
    'Internal',
    'Equal',
    'FermiLevelEqual',
    'FermiLevelEqualElectrode',
    'AppliedVoltage',
    'LocalThermalEquilibrium',
    'LocalThermalEquilibriumContact']


class DirichletBC(BoundaryEquation):
    def evaluate(self, ctx, eq):
        if ctx.solver.poissonOnly and not isinstance(
                self.owner_eq, PoissonEquation):
            return
        yield eq._getDof(), ctx.varsOf(eq.owner_eq)['x'][eq._getIdx()] - self.value(ctx, eq)

    def value(self, ctx, eq):
        raise NotImplementedError()


class DirichletBCWithFunction(DirichletBC):
    def __init__(self, name, function):
        super(DirichletBCWithFunction, self).__init__(name)
        self.function = function

    def initDiscreteEq(self, builder, obj):
        super(DirichletBCWithFunction, self).initDiscreteEq(builder, obj)
        self.function.subInit(self, builder, obj)

    def value(self, ctx, eq):
        return self.function.value(self, ctx, eq)


class AppliedVoltage(DirichletBC):
    def __init__(self, *args, **kwargs):
        calculate_current = kwargs.pop('calculate_current', False)
        super(AppliedVoltage, self).__init__(*args, **kwargs)
        self.calculate_current = calculate_current

    def buildDiscreteEq(self, builder, obj):
        super(AppliedVoltage, self).buildDiscreteEq(builder, obj)

        # register current calculation self.allspecies, to be evaluated between
        # allspecies and alldone checkpoints of owner equation
        obj.allspecies = Funcall(self.allspecies, obj)
        obj.allspecies.depends(obj.owner_eq.allspecies)
        builder.addEvaluation(obj.allspecies)
        obj.owner_eq.alldone.depends(obj.allspecies)
        obj.rs_terminal_name = self.name

    def value(self, ctx, eq):
        if isinstance(ctx.solver, solver.RamoShockleyCalculation):
            return where(self.name == ctx.solver.boundary_name, 1, 0)
        return self.potential(ctx, eq)

    def potential(self, ctx, eq):
        return self.voltage(ctx, eq) - ctx.param(eq, 'workfunction')

    def voltage(self, ctx, eq):
        return ctx.param(eq, 'voltage')

    def allspecies(self, ctx, eq):
        if not ctx.wants_output or ctx.solver.poissonOnly:
            return
        if not self.calculate_current:
            return
        bidx = eq.boundary['bidx']
        J = sum(ctx.varsOf(eq.owner_eq)['Jd_boundary'][bidx])
        for s in eq.owner_eq.species:
            J = J + sum(ctx.varsOf(s)['total_boundary_FdS'][bidx]*s.ze)
        ctx.output([eq, 'Jboundary'], J)


class Zero(DirichletBC):
    def value(self, ctx, eq):
        return 0


class DirichletFromParams(DirichletBC):
    def value(self, ctx, eq):
        return ctx.param(eq.owner_eq, self.name)


class Internal(DirichletBC):
    def __init__(self, other_eq, name):
        super(Internal, self).__init__(name)
        self.name = name
        self.eq_from = other_eq

    def newDiscreteEq(self, builder):
        return builder.newDirichletBC(builder.get(self.owner_eq), self.name,
                                      other_eq_lookup=lambda builder: builder.get(self.eq_from))

    def initDiscreteEq(self, builder, obj):
        super(Internal, self).initDiscreteEq(builder, obj)
        obj.evaluate.depends(builder.get(self.eq_from).load)

        # make sure that this BC will be evaluated before variables of eq_from
        # are freed
        builder.get(self.eq_from).alldone.depends(obj.evaluate)


class Equal(Internal):
    def value(self, ctx, eq):
        return ctx.varsOf(eq.other_eq)['x'][eq._getOtherIdx()]


class FermiLevelEqual(Internal):
    def value(self, ctx, eq):
        if ctx.solver.poissonOnly:
            assert not isinstance(eq, PoissonEquation)
            return
        Effrom = ctx.varsOf(eq.other_eq)['Ef'][eq._getOtherIdx()]
        return self.owner_eq.dos.concentration(
            ctx, eq.owner_eq, eq._getIdx(), Effrom)


class FermiLevelEqualElectrode(DirichletBC):
    F_eps = 1e-10

    def __init__(self, name, image_force=False, **kwargs):
        super(FermiLevelEqualElectrode, self).__init__(name, **kwargs)
        self.image_force = image_force

    def image_correction(self, ctx, eq, ixto):
        if not self.image_force:
            return 0.
        n = eq.mesh.boundaries[self.name]['normal']
        F = eq.z * eq.mesh.dotv(ctx.varsOf(eq.poisson)['Ecellv'][ixto], n)
        F = where(F > self.F_eps, F, self.F_eps)
        return -eq.z * \
            functions.EmtageODwyerBarrierLowering(
                F, getitem(ctx.varsOf(eq.poisson)['epsilon'], ixto))

    def value(self, ctx, eq):
        dos = self.owner_eq.dos
        ixto = eq._getIdx()
        Ef_electrode = -ctx.param(eq, 'voltage') + \
            self.image_correction(ctx, eq.owner_eq, ixto)
        c_electrode = dos.concentration(ctx, eq.owner_eq, ixto, Ef_electrode)
        return c_electrode


LocalThermalEquilibrium = FermiLevelEqual
LocalThermalEquilibriumContact = FermiLevelEqualElectrode
