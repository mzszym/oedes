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

from .charged import ChargedSpecies
from oedes.fvm import DummyConvergenceTest

__all__ = ['FixedCharge']


class FixedCharge(ChargedSpecies):
    "Fixed charge, typically corresponding to ionized dopants"

    def __init__(self, poisson, mesh=None, name=None, z=1, density=None):
        super(FixedCharge, self).__init__(mesh, name, z, poisson=poisson)
        if density is not None:
            self.density = density

    def density(self, x, ctx, eq):
        return ctx.param(eq, 'Nd') - ctx.param(eq, 'Na')

    def buildDiscreteEq(self, ctx, eq):
        super(FixedCharge, self).buildDiscreteEq(ctx, eq)
        eq.convergenceTest = DummyConvergenceTest()

    def load_concentration(self, ctx, eq):
        c = self.density(eq.mesh, ctx, eq)
        ctx.varsOf(eq).update(c=c, j=None)
        self.output_concentration(ctx, eq)
