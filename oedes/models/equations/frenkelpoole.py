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

from .band import MobilityModel
from oedes.ad import where, exp, sqrt, log

__all__ = ['FrenkelPooleMobility']


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
