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

from .source import BulkSource
from oedes.ad import exp, where
import numpy as np
import scipy.constants
from oedes import functions

__all__ = ['OnsagerBraunRecombinationDissociationTerm']


class OnsagerBraunRecombinationDissociationTerm(BulkSource):
    b_eps = 1e-10
    b_max = 100

    def __init__(self, eq, electron_eq, hole_eq,
                 name='dissociation', binding_energy_param=False):
        super(
            OnsagerBraunRecombinationDissociationTerm,
            self).__init__(
                name=name,
                eq_refs=[
                    'eq',
                    'electron_eq',
                    'hole_eq'])
        assert eq.z == electron_eq.z + hole_eq.z, 'inconsistent signs'
        self.eq = eq
        self.electron_eq = electron_eq
        self.hole_eq = hole_eq
        self.binding_energy_param = binding_energy_param

    def evaluate(self, ctx, eq):
        if ctx.solver.poissonOnly:
            return
        assert eq.electron_eq.poisson is eq.hole_eq.poisson
        assert eq.electron_eq.thermal is eq.hole_eq.thermal
        poissonvars = ctx.varsOf(eq.electron_eq.poisson)
        epsilon = poissonvars['epsilon']
        Vt = ctx.varsOf(eq.electron_eq.thermal)['Vt']
        nvars = ctx.varsOf(eq.electron_eq)
        pvars = ctx.varsOf(eq.hole_eq)
        evars = ctx.varsOf(eq.eq)

        if self.binding_energy_param:
            Eb = ctx.param(eq.eq, 'binding_energy')
            a = scipy.constants.elementary_charge / \
                (Eb * 4 * np.pi * epsilon)
        else:
            a = ctx.param(eq.eq, 'distance')
            Eb = scipy.constants.elementary_charge / \
                (4 * np.pi * epsilon * a)
        u = 3. / (4. * np.pi * a**3)  # m^-3
        v = exp(-Eb / Vt)  # 1
        b = (scipy.constants.elementary_charge / (8 * np.pi)) * \
            poissonvars['Ecellm'] / (epsilon * Vt**2)  # 1
        # b=0.
        t = functions.OnsagerFunction(
            where(
                b < self.b_max,
                b,
                self.b_max) +
            self.b_eps)  # 1
        gamma = scipy.constants.elementary_charge * \
            (nvars['mu_cell'] + pvars['mu_cell']) / \
            epsilon  # m^3 /s
        r = gamma * (nvars['c'] * pvars['c'] - ctx.common_param(
            [eq.electron_eq, eq.hole_eq], 'npi'))  # 1/(m^3 s) # TODO
        d = gamma * evars['c'] * u * v * t
        ctx.outputCell([eq.eq, self.name, 'recombination'],
                       r, unit=ctx.units.dconcentration_dt)
        ctx.output([eq.eq, self.name, 'dissociation'],
                   d, unit=ctx.units.dconcentration_dt)
        f = r - d
        self.add(ctx, f, plus=[eq.eq], minus=[eq.hole_eq, eq.electron_eq])
