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

from oedes.ad import exp, where, sqrt
import numpy as np
import scipy.constants


def _v_D(mu, ctx, eq):
    ctx.varsOf(eq)['mu'] = mu
    v = eq.v(eq.mesh.faceaverage(mu), ctx.varsOf(eq.poisson)['E'])
    D = eq.mesh.faceaverage(eq.D(mu, ctx.varsOf(eq.thermal)['Vt']))
    return v, D


def species_v_D_charged_from_params(ctx, eq):
    """
    Default implementation of transported species:
    - assumes constant mobility, takes from parameters
    - assumes diffusion described by Einstein's relation
    Note that mobility is saved for other calculations (such as recombination).
    """
    mu = ctx.param(eq, 'mu')
    if callable(mu):
        mu = mu(
            T=ctx.varsOf(
                eq.thermal)['T'],
            c=ctx.varsOf(eq)['c'],
            E=ctx.varsOf(
                eq.poisson)['Ecellm'])
    return _v_D(mu, ctx, eq)


def species_v_D_FrenkelPoole(ctx, eq):
    mu0 = ctx.param(eq, 'mu0')
    gamma = ctx.param(eq, 'gamma')
    v = gamma * sqrt(ctx.varsOf(eq.poisson)['Ecellm'] + 1e-10)
    v_max = np.log(1e10)
    v_min = -v_max
    v = where(v < v_max, v, v_max)
    v = where(v > v_min, v, v_min)
    mu = mu0 * exp(v)
    return _v_D(mu, ctx, eq)


def species_v_D_not_transported(ctx, eq):
    "Not transported species"
    return None


def simple_doping(ctx, eq):
    return (ctx.param(eq, ctx.UPPER, 'Nd') - ctx.param(eq,
                                                       ctx.UPPER, 'Na')) * scipy.constants.elementary_charge


def species_v_D_limited_upwind(ctx, eq):
    v, D = species_v_D_charged_from_params(ctx, eq)
    c = ctx.varsOf(eq)['c']
    c = where(v > 0, c[eq.mesh.faces['+']], c[eq.mesh.faces['-']])
    v = v * (1. - c / ctx.param(eq, 'N0'))
    return v, D
