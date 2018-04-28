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

__all__ = ['DirectGeneration', 'SimpleGenerationTerm', 'SimpleDecayTerm']


class DirectGeneration(BulkSource):
    def __init__(self, eqs, function, name='absorption'):
        super(
            DirectGeneration,
            self).__init__(
            name=name,
            eq_reflists=['eqs'])
        assert sum(eq.z for eq in eqs) == 0
        self.function = function
        self.eqs = eqs
        self.G = self.function(self.eqs[0].mesh.cells['center'])

    def evaluate(self, ctx, eq):
        if ctx.solver.poissonOnly:
            return
        g = self.G * ctx.param(eq, 'I')
        ctx.output([eq, 'G'], g)
        self.add(ctx, g, plus=eq.eqs)


class SimpleGenerationTerm(DirectGeneration):
    def __init__(self, eq, function, name='absorption'):
        super(
            SimpleGenerationTerm,
            self).__init__(
            [eq],
            function,
            name=eq.name +
            '.' +
            name)


class SimpleDecayTerm(BulkSource):
    def __init__(self, eq, name='decay'):
        super(SimpleDecayTerm, self).__init__(name=name, eq_refs=['eq'])
        assert eq.z == 0
        self.eq = eq

    def evaluate(self, ctx, eq):
        if ctx.solver.poissonOnly:
            return
        f = ctx.varsOf(eq.eq)['c'] * ctx.param(eq.eq, self.name)
        self.add(ctx, f, minus=[eq.eq])
