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

from .base import Calculation, Funcall
import scipy.constants
from oedes.ad import sum
from oedes.models import solver
from collections import defaultdict

__all__ = ['ConstTemperature', 'NewCurrentCalculation']


class ConstTemperature(Calculation):
    def build(self, builder):
        obj = super(ConstTemperature, self).build(builder)
        obj.load = Funcall(self.load, obj)
        builder.addEvaluation(obj.load)
        return obj

    def load(self, ctx, eq):
        if ctx.solver.poissonOnly:
            return
        T = ctx.param(eq, 'T')
        Vt = T * scipy.constants.Boltzmann / scipy.constants.elementary_charge
        ctx.varsOf(eq).update(dict(T=T, Vt=Vt))


class RamoShockleyCurrentCalculation(Calculation):
    def __init__(self, poisson_eqs, name=None):
        super(RamoShockleyCurrentCalculation, self).__init__(name)
        self.poisson_eqs = poisson_eqs

    def build(self, builder):
        obj = super(RamoShockleyCurrentCalculation, self).build(builder)
        obj.evaluate = Funcall(self.evaluate, obj)
        builder.addInitializer(self._init, obj)
        builder.addEvaluation(obj.evaluate)
        return obj

    def _init(self, builder, obj):
        obj.poisson_eqs = tuple(map(builder.get, self.poisson_eqs))
        for p in obj.poisson_eqs:
            obj.evaluate.depends(p.allspecies)

    def evaluate(self, ctx, eq):
        if not ctx.wants_output:
            return
        if ctx.solver.poissonOnly:
            return
        d = defaultdict(int)
        default_electrodes = set()
        for p in eq.poisson_eqs:
            Jdict = ctx.varsOf(p)['Jramo_shockley']
            for k in Jdict.keys():
                d[k] = d[k] + Jdict[k]
            if p.mesh.default_electrode is not None:
                default_electrodes.add(p.mesh.default_electrode)
        for k, v in d.items():
            ctx.output([eq, k, 'J'], v)
        if len(eq.poisson_eqs) == 1:
            if default_electrodes and any(k in d for k in default_electrodes):
                Jdefault = 0
                for k in default_electrodes:
                    if k in d:
                        Jdefault = Jdefault + d[k]
                ctx.output([eq, 'J'], Jdefault)


NewCurrentCalculation = RamoShockleyCurrentCalculation  # TODO
