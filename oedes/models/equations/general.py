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

__all__ = [
    'AdvectionDiffusion',
    'species_v_D_charged_from_params',
    'species_v_D_FrenkelPoole',
    'species_v_D_limited']

from oedes.ad import exp, where, sqrt
import numpy as np
from .transport import TransportedSpecies
from .frenkelpoole import FrenkelPooleMobility


class AdvectionDiffusion(TransportedSpecies):
    "General advection diffusion equation"

    def __init__(self, *args, **kwargs):
        v_D = kwargs.pop('v_D', None)
        super(AdvectionDiffusion, self).__init__(*args, **kwargs)
        self.v_D = v_D

    def evaluate_mobility(self, ctx, eq):
        ctx.varsOf(eq)['v_D'] = self.v_D(ctx, eq)


def species_v_D_charged_from_params(ctx, eq):
    """
    Simple transport model:
    - assumes diffusion described by Einstein's relation
    - mobility given as parameter (mu), or as a function f(T,c,E)
    """
    mu = ctx.param(eq, 'mu')
    if callable(mu):
        mu_cell = mu(
            T=ctx.varsOf(
                eq.thermal)['T'],
            c=ctx.varsOf(eq)['c'],
            E=ctx.varsOf(
                eq.poisson)['Ecellm'])
        mu = eq.mesh.faceaverage(mu_cell)
        ctx.varsOf(eq).update(mu_face=mu, mu_cell=mu_cell)
    else:
        ctx.varsOf(eq).update(mu_face=mu, mu_cell=mu)
    v = eq.v(mu, ctx.varsOf(eq.poisson)['E'])
    D = eq.D(mu, ctx.varsOf(eq.thermal)['Vt'])
    return v, D


def species_v_D_FrenkelPoole(ctx, eq):
    "Frenkel - Poole mobility model"
    FrenkelPooleMobility().mobility(None, ctx, eq)
    mu_face = ctx.varsOf(eq)['mu_face']
    mu_cell = ctx.varsOf(eq)['mu_cell']
    v = eq.v(mu_face, ctx.varsOf(eq.poisson)['E'])
    D = eq.mesh.faceaverage(eq.D(mu_cell, ctx.varsOf(eq.thermal)['Vt']))
    return v, D


def species_v_D_limited(ctx, eq):
    "Species with concentration limitation"
    v, D = species_v_D_charged_from_params(ctx, eq)
    c = ctx.varsOf(eq)['c']
    c = where(v > 0, c[eq.mesh.faces['+']], c[eq.mesh.faces['-']])
    v = v * (1. - c / ctx.param(eq, 'N0'))
    return v, D
