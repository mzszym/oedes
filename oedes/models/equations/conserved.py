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
from .charged import ChargedSpecies
from oedes.fvm import ElementwiseConvergenceTest

__all__ = ['ConservedSpecies', 'NontransportedSpecies']


class ConservedSpecies(ChargedSpecies, ConservationEquation):
    """
    Species with charge, and satisfying conservation equation

    Outputs:
    --------
    j, jdrift, jdiff : face variable
        total flux density, drift part of flux density, diffusion part of flux density  [ 1/(m^2 s)]
    J, Jdrift, Jdiff : face variable
        total current density, drift part of current density, diffusion part of current density  [ 1/(m^2 s)]
    """

    def __init__(self, *args, **kwargs):
        super(ConservedSpecies, self).__init__(*args, **kwargs)
        self.convergenceTest = ElementwiseConvergenceTest(
            atol=5e7, rtol=1e-7, eps=1e-10)
        self.defaultBCConvergenceTest = ElementwiseConvergenceTest(
            atol=1., rtol=1e-7)

    def newDiscreteEq(self, builder):
        return builder.newTransportChargedEquation(
            builder.getMesh(self.mesh), self.name, self.z)

    def load_concentration(self, ctx, eq):
        variables = ctx.varsOf(eq)
        c = variables['x']
        ct = variables['xt']
        variables.update(dict(c=c, ct=ct))
        self.output_concentration(ctx, eq)

    def load(self, ctx, eq):
        super(ConservedSpecies, self).load(ctx, eq)
        if ctx.solver.poissonOnly:
            return
        ctx.varsOf(eq).update(sources=0)
        assert eq.poisson.mesh is eq.mesh, 'multiple meshes currently unsupported'

    def calculate_fluxes(self, ctx, eq, v, D):
        c = ctx.varsOf(eq)['c']
        if not ctx.wants_output:
            return eq.faceflux(c=c, v=v, D=D, full_output=False)
        f = eq.faceflux(c=c, v=v, D=D, full_output=True)
        ctx.outputFace([eq, 'j'], f['flux'], unit=ctx.units.flux)
        ctx.outputFace([eq, 'jdrift'], f['flux_v'],
                       unit=ctx.units.flux)
        ctx.outputFace([eq, 'jdiff'], f['flux_D'], unit=ctx.units.flux)
        ctx.outputFace([eq, 'J'], f['flux']*eq.ze,
                       unit=ctx.units.current_density)
        ctx.outputFace([eq, 'Jdrift'], f['flux_v']*eq.ze,
                       unit=ctx.units.current_density)
        ctx.outputFace([eq, 'Jdiff'], f['flux_D']*eq.ze,
                       unit=ctx.units.current_density)
        return f['flux']

    def evaluate_fluxes(self, ctx, eq):
        raise NotImplementedError()

    def evaluate(self, ctx, eq):
        if ctx.solver.poissonOnly:
            return self.identity(ctx, eq)
        variables = ctx.varsOf(eq)
        j = self.evaluate_fluxes(ctx, eq)
        variables.update(j=j)
        return self.residuals(
            ctx, eq, flux=j, source=variables['sources'], transient=variables['ct'])


class NontransportedSpecies(ConservedSpecies):
    def evaluate_fluxes(self, ctx, eq):
        return None
