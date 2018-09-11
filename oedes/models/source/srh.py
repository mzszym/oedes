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

from .source import _Recombination
from oedes.ad import exp

__all__ = ['SRH']


class SRH(_Recombination):
    "SRH (steady-state) term"

    def __init__(self, *args, **kwargs):
        name = kwargs.pop('name', 'srh')
        super(SRH, self).__init__(*args, name=name, **kwargs)

    def evaluate(self, ctx, eq):
        assert eq.hole_eq.thermal is eq.electron_eq.thermal
        if ctx.solver.poissonOnly:
            return
        Vt = ctx.varsOf(eq.electron_eq.thermal)['Vt']
        Et = ctx.param(eq, 'energy')
        ni = ctx.param(eq.electron_eq, 'N0') * \
            exp((Et - ctx.param(eq.electron_eq, 'energy')) / Vt)
        pi = ctx.param(eq.hole_eq, 'N0') * \
            exp((ctx.param(eq.hole_eq, 'energy') - Et) / Vt)
        Cn = ctx.param(eq.electron_eq, self.name, 'trate')
        Cp = ctx.param(eq.hole_eq, self.name, 'trate')
        n = ctx.varsOf(eq.electron_eq)['c']
        p = ctx.varsOf(eq.hole_eq)['c']
        g = Cn * Cp * ctx.param(eq, 'N0') * \
            (n * p - ni * pi) / (Cn * (n + ni) + Cp * (p + pi))
        ctx.outputCell([eq, 'G'],  g, mesh=eq.electron_eq.mesh,
                       unit=ctx.units.dconcentration_dt)
        self.add(ctx, g, minus=[eq.electron_eq, eq.hole_eq])
