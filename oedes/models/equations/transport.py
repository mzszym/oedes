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

from oedes.utils import Funcall
from .conserved import ConservedSpecies

__all__ = ['TransportedSpecies']


class TransportedSpecies(ConservedSpecies):
    def evaluate_fluxes(self, ctx, eq):
        v_D = ctx.varsOf(eq)['v_D']
        if v_D is None:
            return None
        return self.calculate_fluxes(ctx, eq, *v_D)

    def buildDiscreteEq(self, builder, obj):
        super(TransportedSpecies, self).buildDiscreteEq(builder, obj)
        obj.mobility = Funcall(self.mobility, obj, depends=[obj.load])
        obj.evaluate.depends(obj.mobility)
        builder.addEvaluation(obj.mobility)

    def mobility(self, ctx, eq):
        if ctx.solver.poissonOnly:
            return
        self.evaluate_mobility(ctx, eq)

    def evaluate_mobility(self, ctx, eq):
        raise NotImplementedError()
