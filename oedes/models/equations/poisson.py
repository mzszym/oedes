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

from .base import ConservationEquation, DelegateConvergenceTest, Funcall
from oedes.fvm import FVMPoissonEquation, ElementwiseConvergenceTest
from oedes.ad import dot, sum, getitem
import scipy.constants
from oedes.models import solver
from oedes.utils import WeakList

__all__ = ['Poisson']


class Poisson(ConservationEquation):
    def __init__(self, mesh, name='poisson'):
        super(Poisson, self).__init__(mesh=mesh, name=name)
        self.convergenceTest = ElementwiseConvergenceTest(
            atol=1e-15, rtol=1e-7)
        self.defaultBCConvergenceTest = ElementwiseConvergenceTest(
            atol=1e-15, rtol=1e-7)
        self.additional_charge = []

    def build(self, builder):
        obj = builder.newPoissonEquation(builder.getMesh(self.mesh), self.name)
        obj.convergenceTest = DelegateConvergenceTest(self)
        # TODO: refactor so this is unnecessary (for convergence test)
        obj.prefix = self.prefix
        obj.load = Funcall(self.load, obj)
        obj.evaluate = Funcall(self.evaluate, obj, depends=[obj.load])
        obj.allspecies = Funcall(self.allspecies, obj, depends=[obj.evaluate])
        obj.ramo_shockley_potentials = dict()
        builder.addEvaluation(obj.load)
        builder.addEvaluation(obj.evaluate)
        builder.addEvaluation(obj.allspecies)
        obj.species = WeakList()
        builder.add(self, obj)
        return obj

    def load(self, ctx, eq):
        super(Poisson, self).load(ctx, eq)
        newvars = ctx.varsOf(eq)
        potential = newvars['x']
        newvars['potential'] = potential
        E = eq.E(potential)
        newvars['E'] = E
        newvars['Et'] = eq.E(newvars['xt'])
        if isinstance(ctx.solver, solver.RamoShockleyCalculation):
            epsilon = scipy.constants.epsilon_0
        else:
            epsilon = scipy.constants.epsilon_0 * \
                ctx.param(eq, ctx.UPPER, 'epsilon_r')
        newvars['epsilon'] = epsilon
        Ecellv = eq.mesh.cellaveragev(E)
        newvars['Ecellv'] = Ecellv
        newvars['Ecellm'] = eq.mesh.magnitudev(Ecellv)

    def evaluate(self, ctx, eq):
        "Part of evaluate dealing with Poisson's equation"
        assert isinstance(eq, FVMPoissonEquation)
        newvars = ctx.varsOf(eq)
        epsilon = newvars['epsilon']
        potential = newvars['potential']
        total_charge_density = 0
        if not isinstance(ctx.solver, solver.RamoShockleyCalculation):
            for s in eq.species:
                total_charge_density = total_charge_density + \
                    s.ze * ctx.varsOf(s)['c']
            for f in self.additional_charge:
                total_charge_density = total_charge_density + f(ctx, eq)
        elif ctx.solver.store:
            eq.ramo_shockley_potentials[ctx.solver.boundary_name] = newvars['potential']
        faceepsilon = eq.mesh.faceaverage(epsilon)
        E = newvars['E']
        Et = newvars['Et']
        D = eq.displacement(E, faceepsilon)
        Dt = eq.displacement(Et, faceepsilon)
        newvars['Dt'] = Dt
        newvars['D'] = D
        ctx.outputCell([eq, ctx.UPPER, 'poisson.total_charge_density'],
                       total_charge_density, unit=ctx.units.charge_density)  # TODO
        ctx.outputCell([eq, ctx.UPPER, 'potential'],
                       potential, unit=ctx.units.potential)
        ctx.outputFace([eq, ctx.UPPER, 'E'], E, unit=ctx.units.electric_field)
        ctx.outputFace([eq, ctx.UPPER, 'Et'], Et)
        ctx.outputFace([eq, ctx.UPPER, 'D'], D)
        ctx.outputFace([eq, ctx.UPPER, 'Dt'], Dt,
                       unit=ctx.units.current_density)
        yield eq.residuals(eq.mesh.internal, D, cellsource=total_charge_density)
        FdS_boundary = dot(eq.mesh.boundary.fluxsum, D)
        newvars['Jd_boundary'] = dot(eq.mesh.boundary.fluxsum, Dt)
        self.evaluate_bc(ctx, eq, potential, D, FdS_boundary,
                         cellsource_boundary=getitem(total_charge_density, eq.mesh.boundary.idx))

    def allspecies(self, ctx, eq):
        if not ctx.wants_output or ctx.solver.poissonOnly:
            return
        result = dict()
        for name, phi in eq.ramo_shockley_potentials.items():
            w = eq.mesh.facegrad(
                phi) * eq.mesh.faces['dr'] * eq.mesh.faces['surface']
            J = sum(w * ctx.varsOf(eq)['Dt'])
            for s in eq.species:
                j = ctx.varsOf(s)['j']
                if j is None:
                    continue
                J = J + sum(w * j * s.ze)
            result[name] = J
        ctx.varsOf(eq)['Jramo_shockley'] = result
