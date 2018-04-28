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

from .base import ConservationEquation
from oedes.fvm import FVMPoissonEquation, ElementwiseConvergenceTest
from oedes.ad import dot, sum, getitem
import scipy.constants
from oedes.models import solver
from oedes.utils import WeakList, Funcall

__all__ = ['PoissonEquation', 'Poisson']


class PoissonEquation(ConservationEquation):
    """
    Poisson's equation

    Checkpoint allspecies is run after currents for all species inside the domain are calculated
    Checkpoint alldone is the last before freeing variables
    """

    def __init__(self, *args, **kwargs):
        super(Poisson, self).__init__(*args, **kwargs)
        self.convergenceTest = ElementwiseConvergenceTest(
            atol=1e-15, rtol=1e-7)
        self.defaultBCConvergenceTest = ElementwiseConvergenceTest(
            atol=1e-15, rtol=1e-7)

    def newDiscreteEq(self, builder):
        return builder.newPoissonEquation(
            builder.getMesh(self.mesh), self.name)

    def buildDiscreteEq(self, builder, obj):
        super(Poisson, self).buildDiscreteEq(builder, obj)
        obj.species = WeakList()
        obj.allspecies = Funcall(self.allspecies, obj, depends=[obj.evaluate])
        obj.alldone = Funcall(None, obj, depends=[obj.allspecies])
        obj.ramo_shockley_potentials = dict()
        builder.addEvaluation(obj.allspecies)

    def load(self, ctx, eq):
        super(Poisson, self).load(ctx, eq)
        variables = ctx.varsOf(eq)
        potential = variables['x']
        E = eq.E(potential)
        Et = eq.E(variables['xt'])
        if isinstance(ctx.solver, solver.RamoShockleyCalculation):
            epsilon = scipy.constants.epsilon_0
        else:
            epsilon = scipy.constants.epsilon_0 * \
                ctx.param(eq, 'epsilon_r')
        Ecellv = eq.mesh.cellaveragev(E)
        Ecellm = eq.mesh.magnitudev(Ecellv)
        variables.update(
            Ecellm=Ecellm,
            Ecellv=Ecellv,
            epsilon=epsilon,
            E=E,
            Et=Et,
            potential=potential)

    def evaluate(self, ctx, eq):
        "Part of evaluate dealing with Poisson's equation"
        assert isinstance(eq, FVMPoissonEquation)
        variables = ctx.varsOf(eq)
        total_charge_density = 0
        if not isinstance(ctx.solver, solver.RamoShockleyCalculation):
            for s in eq.species:
                total_charge_density = total_charge_density + \
                    s.ze * ctx.varsOf(s)['c']
        elif ctx.solver.store:
            eq.ramo_shockley_potentials[ctx.solver.boundary_name] = variables['potential']
        epsilon = variables['epsilon']
        faceepsilon = eq.mesh.faceaverage(epsilon)
        potential = variables['potential']
        E = variables['E']
        Et = variables['Et']
        D = eq.displacement(E, faceepsilon)
        Dt = eq.displacement(Et, faceepsilon)
        variables.update(D=D, Dt=Dt)
        ctx.outputCell([eq, 'total_charge_density'],
                       total_charge_density, unit=ctx.units.charge_density)
        ctx.outputCell([eq, 'potential'],
                       potential, unit=ctx.units.potential)
        ctx.outputFace([eq, 'E'], E, unit=ctx.units.electric_field)
        ctx.outputFace([eq, 'Et'], Et)
        ctx.outputFace([eq, 'D'], D)
        ctx.outputFace([eq, 'Dt'], Dt, unit=ctx.units.current_density)
        variables['Jd_boundary'] = dot(eq.mesh.boundary.fluxsum, Dt)
        return self.residuals(ctx, eq, flux=D, source=total_charge_density)

    def allspecies(self, ctx, eq):
        if not ctx.wants_output or ctx.solver.poissonOnly:
            return
        result = dict()
        for name, phi in eq.ramo_shockley_potentials.items():
            w = eq.mesh.facegrad(
                phi) * eq.mesh.faces['dr'] * eq.mesh.faces['surface']
            J = sum(w * ctx.varsOf(eq)['Dt'])
            for s in eq.species:
                svars = ctx.varsOf(s)
                j = ctx.varsOf(s)['j']
                if j is None:
                    continue
                J = J + sum(w * j * s.ze)
            result[name] = J
        ctx.varsOf(eq)['Jramo_shockley'] = result


Poisson = PoissonEquation
