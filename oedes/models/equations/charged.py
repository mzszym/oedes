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

import scipy.constants
from oedes.models import solver
from .base import EquationWithMesh
from .poisson import Poisson

__all__ = ['ChargedSpecies']


class ChargedSpecies(EquationWithMesh):
    """
    Abstract class for any distribution of charge
    """

    def __init__(self, mesh=None, name=None, z=1, poisson=None, thermal=None):
        if poisson is not None and mesh is None:
            mesh = poisson.mesh
        super(ChargedSpecies, self).__init__(mesh, name)
        self.z = z
        self.poisson = poisson
        self.thermal = thermal

    def newDiscreteEq(self, builder):
        obj = super(ChargedSpecies, self).newDiscreteEq(builder)
        obj.ze = scipy.constants.elementary_charge * self.z
        obj.z = self.z
        obj.mesh = builder.getMesh(self.mesh)
        return obj

    def buildDiscreteEq(self, builder, obj):
        super(ChargedSpecies, self).buildDiscreteEq(builder, obj)
        assert isinstance(self.poisson, Poisson)

    def initDiscreteEq(self, builder, obj):
        super(ChargedSpecies, self).initDiscreteEq(builder, obj)
        obj.poisson = builder.get(self.poisson)
        obj.poisson.species.append(obj)
        obj.poisson.evaluate.depends(obj.load)
        obj.poisson.allspecies.depends(obj.evaluate)
        obj.load.depends(obj.poisson.load)
        obj.alldone.depends(obj.poisson.load)  # keep vars
        obj.alldone.depends(obj.poisson.alldone)
        if self.thermal is not None:
            obj.thermal = builder.get(self.thermal)
            obj.load.depends(obj.thermal.load)
            obj.alldone.depends(obj.thermal.load)  # keep vars

    def load(self, ctx, eq):
        super(ChargedSpecies, self).load(ctx, eq)
        if isinstance(ctx.solver, solver.RamoShockleyCalculation):
            return
        self.load_concentration(ctx, eq)

    def load_concentration(self, ctx, eq):
        raise NotImplementedError()

    def output_concentration(self, ctx, eq):
        variables = ctx.varsOf(eq)
        c = variables['c']
        ctx.outputCell([eq, 'c'], c, unit=ctx.units.concentration)
        ctx.outputCell([eq, 'charge'], c*eq.ze, unit=ctx.units.charge_density)
        if 'ct' in variables:
            ctx.output([eq, 'ct'], variables['ct'],
                       unit=ctx.units.dconcentration_dt)
