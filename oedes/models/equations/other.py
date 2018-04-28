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
import scipy.constants
from oedes.ad import sum
from oedes.models import solver
from collections import defaultdict
from oedes.utils import ArgStack
from oedes.models.solver import RamoShockleyCalculation
from oedes import solve
import numpy as np

__all__ = ['ConstTemperature', 'RamoShockleyCurrentCalculation']


class ConstTemperature(Calculation):
    def load(self, ctx, eq):
        if isinstance(ctx.solver, solver.RamoShockleyCalculation):
            return
        T = ctx.param(eq, 'T')
        Vt = T * scipy.constants.Boltzmann / scipy.constants.elementary_charge
        ctx.newVars(eq).update(dict(T=T, Vt=Vt))


class RamoShockleyCurrentCalculation(Calculation):
    def __init__(self, poisson_eqs, name=None):
        super(RamoShockleyCurrentCalculation, self).__init__(name)
        self.poisson_eqs = poisson_eqs

    def initDiscreteEq(self, builder, obj):
        super(
            RamoShockleyCurrentCalculation,
            self).initDiscreteEq(
            builder,
            obj)
        obj.poisson_eqs = tuple(map(builder.get, self.poisson_eqs))
        for p in obj.poisson_eqs:
            obj.evaluate.depends(p.allspecies)
        builder.registerFinalizer(self.generateTestfunctions)

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

    @classmethod
    def generateTestfunctions(
            cls, builder, model, dtype=np.double, solve_kwargs=None):
        if solve_kwargs is None:
            solve_kwargs = dict()
        rs_boundaries = set()
        for eq in builder.equations:
            if hasattr(eq, 'rs_terminal_name'):
                rs_boundaries.add(eq.rs_terminal_name)
        params = dict()
        for b in rs_boundaries:
            s = RamoShockleyCalculation(b, store=False)
            X = np.asarray(model.X, dtype=dtype)
            x = solve(model, X, params, niter=1, solver=s, **solve_kwargs)
            s = RamoShockleyCalculation(b, store=True)
            model.output(
                0,
                x,
                np.zeros_like(x),
                params,
                solver=s,
                **solve_kwargs)
