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

from oedes.utils import Calculation
from oedes import functions
from oedes.models.equations import TransportedSpecies

__all__ = ['BulkSource', 'LangevinRecombination', 'DirectRecombination']


class BulkSource(Calculation):
    def __init__(self, name, eq_refs=[], eq_reflists=[]):
        super(BulkSource, self).__init__(name=name)
        self._eq_refs = eq_refs
        self._eq_reflists = eq_reflists

    def add(self, ctx, r, plus=[], minus=[]):
        z = 0
        for p in plus:
            pvars = ctx.varsOf(p)
            pvars['sources'] = pvars['sources'] + r
            z = z + p.z
        for n in minus:
            nvars = ctx.varsOf(n)
            nvars['sources'] = nvars['sources'] - r
            z = z - n.z
        assert z == 0, 'not charge neutral'

    def initDiscreteEq(self, builder, obj):
        super(BulkSource, self).initDiscreteEq(builder, obj)

        for k in self._eq_refs:
            eq = builder.get(getattr(self, k))
            obj.evaluate.depends(eq.load)
            eq.evaluate.depends(obj.evaluate)
            setattr(obj, k, eq)
        for k in self._eq_reflists:
            eqs = tuple(map(builder.get, getattr(self, k)))
            for eq in eqs:
                obj.evaluate.depends(eq.load)
                eq.evaluate.depends(obj.evaluate)
            setattr(obj, k, eqs)


class _Recombination(BulkSource):
    "Langevinian recombination term, to be used"

    def __init__(self, *args, **kwargs):
        output_name = kwargs.pop('output_name', 'R')
        name = kwargs.pop('name', None)
        self.skip_intrinsic = kwargs.pop('skip_intrinsic', False)
        if len(args) == 2:
            electron_eq, hole_eq = args
            self.semiconductor = None
        else:
            self.semiconductor, = args
            electron_eq, = self.semiconductor.electron
            hole_eq, = self.semiconductor.hole

        super(
            _Recombination,
            self).__init__(
            name,
            eq_refs=[
                'electron_eq',
                'hole_eq'], **kwargs)
        assert electron_eq.z == -1
        assert hole_eq.z == 1
        assert hole_eq.poisson is electron_eq.poisson

        self.electron_eq = electron_eq
        self.hole_eq = hole_eq
        self.output_name = output_name

    def initDiscreteEq(self, builder, eq):
        super(_Recombination, self).initDiscreteEq(builder, eq)
        if self.semiconductor is not None:
            eq.semiconductor = builder.get(self.semiconductor)
            eq.semiconductor.alldone.depends(eq.evaluate)
            eq.evaluate.depends(eq.semiconductor.load)
        else:
            eq.semiconductor = None

    def intrinsic_from_semiconductor(self, ctx, eq):
        svars = ctx.varsOf(eq.semiconductor)
        return svars['conc_Ef'][id(eq.hole_eq)] * \
            svars['conc_Ef'][id(eq.electron_eq)]

    def evaluate(self, ctx, eq):
        if ctx.solver.poissonOnly:
            return
        R = self.evaluate_recombination(ctx, eq)
        ctx.outputCell([eq,
                        self.output_name],
                       R,
                       mesh=eq.electron_eq.mesh,
                       unit=ctx.units.dconcentration_dt)
        self.add(ctx, R, minus=[eq.electron_eq, eq.hole_eq])

    def intrinsic_from_params(self, ctx, eq):
        return ctx.common_param([eq.electron_eq, eq.hole_eq], 'npi')

    def intrinsic(self, ctx, eq):
        if self.skip_intrinsic:
            return 0.
        if eq.semiconductor is not None:
            return self.intrinsic_from_semiconductor(ctx, eq)
        else:
            return self.intrinsic_from_params(ctx, eq)


class LangevinRecombination(_Recombination):
    def evaluate_recombination(self, ctx, eq):
        nvars = ctx.varsOf(eq.electron_eq)
        pvars = ctx.varsOf(eq.hole_eq)
        if not isinstance(self.electron_eq, TransportedSpecies):
            nvars = dict(c=nvars['c'], mu_cell=0.)
        if not isinstance(self.hole_eq, TransportedSpecies):
            pvars = dict(c=pvars['c'], mu_cell=0.)
        epsilon = ctx.varsOf(eq.electron_eq.poisson)['epsilon']
        return functions.LangevinRecombination(nvars['mu_cell'], pvars['mu_cell'], nvars['c'], pvars['c'],
                                               epsilon, self.intrinsic(ctx, eq))


class DirectRecombination(_Recombination):
    def __init__(self, *args, **kwargs):
        self.param_name = kwargs.pop('param_name', 'beta')
        super(DirectRecombination, self).__init__(*args, **kwargs)

    def evaluate_recombination(self, ctx, eq):
        return ctx.param(eq, self.param_name) * \
            (ctx.varsOf(eq.electron_eq)['c'] *
             ctx.varsOf(eq.hole_eq)['c'] -
             self.intrinsic(ctx, eq))
