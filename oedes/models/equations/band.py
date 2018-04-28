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

__all__ = ['BandTransport', 'MobilityModel', 'MobilityFromParams']

from .boltzmann import BoltzmannDOS
from .transport import TransportedSpecies
from .dos import WithDOS
from oedes.utils import SubEquation
from oedes.ad import where, exp, sqrt, log


class MobilityModel(SubEquation):
    def mobility(self, parent, ctx, parent_eq):
        "Sets mu_face, mu_cell variables. Normally mu_face is used for transport, mu_cell for some other calculations (ie. recombination)"
        raise NotImplementedError()


class MobilityFromParams(MobilityModel):
    def mobility(self, parent, ctx, eq):
        mu = ctx.param(eq, 'mu')
        ctx.varsOf(eq).update(mu_face=mu, mu_cell=mu)


class FrenkelPooleMobility(MobilityModel):
    def mobility(self, parent, ctx, eq):
        mu0 = ctx.param(eq, 'mu0')
        gamma = ctx.param(eq, 'gamma')
        v = gamma * sqrt(ctx.varsOf(eq.poisson)['Ecellm'] + 1e-10)
        v_max = log(1e10)
        v_min = -v_max
        v = where(v < v_max, v, v_max)
        v = where(v > v_min, v, v_min)
        mu = mu0 * exp(v)
        mu_face = eq.mesh.faceaverage(mu)
        ctx.varsOf(eq)['mu_face'] = mu_face
        ctx.varsOf(eq)['mu_cell'] = mu


class BandTransport(WithDOS, TransportedSpecies):
    alternative = False

    def __init__(self, mesh=None, name=None, z=1, poisson=None,
                 thermal=None, dos=None, mobility_model=None):
        assert z in [1, -1]
        super(
            BandTransport,
            self).__init__(
            mesh=mesh,
            name=name,
            z=z,
            poisson=poisson,
            thermal=thermal)
        if dos is None:
            dos = BoltzmannDOS()
        if mobility_model is None:
            mobility_model = MobilityFromParams()
        self.dos = dos
        self.mobility_model = mobility_model

    def evaluate_mobility(self, ctx, eq):
        self.mobility_model.mobility(self, ctx, eq)

    def initDiscreteEq(self, builder, obj):
        super(BandTransport, self).initDiscreteEq(builder, obj)
        self.mobility_model.subInit(self, builder, obj)

    def evaluate_fluxes(self, ctx, eq):
        mu = ctx.varsOf(eq)['mu_face']
        if self.alternative:
            Ef_correction = self.dos.Ef_correction(ctx, eq)
            v = eq.v(
                mu,
                ctx.varsOf(
                    eq.poisson)['E'] -
                eq.mesh.facegrad(Ef_correction) *
                eq.z)
            D = eq.D(mu, ctx.varsOf(eq.thermal)['Vt'])
        else:
            D_mu = self.dos.D_mu(self, ctx, eq)
            v = eq.v(mu, ctx.varsOf(eq.poisson)['E'])
            D = eq.D(mu, D_mu)
        return self.calculate_fluxes(ctx, eq, v, D)
