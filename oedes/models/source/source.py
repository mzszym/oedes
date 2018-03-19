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

from oedes.models.equations import Calculation, Funcall
from oedes import functions

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

    def build(self, builder):
        obj = super(BulkSource, self).build(builder)
        obj.evaluate = Funcall(self.evaluate, obj)
        builder.addInitializer(self._init, obj)
        builder.addEvaluation(obj.evaluate)
        return obj

    def _init(self, builder, obj):
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

    def __init__(self, electron_eq, hole_eq, output_name='R', name=None):
        super(
            _Recombination,
            self).__init__(
            name,
            eq_refs=[
                'electron_eq',
                'hole_eq'])
        assert electron_eq.z == -1
        assert hole_eq.z == 1
        assert hole_eq.poisson is electron_eq.poisson

        self.electron_eq = electron_eq
        self.hole_eq = hole_eq
        self.output_name = output_name

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


class LangevinRecombination(_Recombination):
    def evaluate_recombination(self, ctx, eq):
        nvars = ctx.varsOf(eq.electron_eq)
        pvars = ctx.varsOf(eq.hole_eq)
        epsilon = ctx.varsOf(eq.electron_eq.poisson)['epsilon']
        return functions.LangevinRecombination(nvars['mu'], pvars['mu'], nvars['c'], pvars['c'],
                                               epsilon, ctx.common_param([eq.electron_eq, eq.hole_eq], 'npi'))


class DirectRecombination(_Recombination):
    def __init__(self, *args, **kwargs):
        self.param_name = kwargs.pop('param_name', 'beta')
        super(DirectRecombination, self).__init__(*args, **kwargs)

    def evaluate_recombination(self, ctx, eq):
        return ctx.param(eq, self.param_name) * \
            (ctx.varsOf(eq.electron_eq)['c'] *
             ctx.varsOf(eq.hole_eq)['c'] -
             ctx.common_param([eq.electron_eq, eq.hole_eq], 'npi'))
