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

__all__ = ['OhmicContact', 'OhmicBCPoisson', 'OhmicBCSpecies']

from .boundary import AppliedVoltage, DirichletBC
import weakref
from oedes.ad import getitem
from oedes.utils import Coupled
from oedes.models.equations import WithDOS


class OhmicBCPoisson(AppliedVoltage):
    def __init__(self, name, semiconductor, poisson, **kwargs):
        super(OhmicBCPoisson, self).__init__(name, **kwargs)
        self.semiconductor = semiconductor
        self._owner_eq_weak = weakref.ref(poisson)

    def initDiscreteEq(self, builder, eq):
        super(OhmicBCPoisson, self).initDiscreteEq(builder, eq)
        eq.semiconductor = builder.get(self.semiconductor)
        eq.semiconductor.alldone.depends(eq.evaluate)
        eq.evaluate.depends(eq.semiconductor.load)

    def potential(self, ctx, eq):
        return self.voltage(
            ctx, eq) + getitem(ctx.varsOf(eq.semiconductor)['Efv'], eq._getIdx())


class OhmicBCSpecies(DirichletBC):
    def __init__(self, name, semiconductor, species):
        super(OhmicBCSpecies, self).__init__(name)
        self.semiconductor = semiconductor
        self._owner_eq_weak = weakref.ref(species)

    def initDiscreteEq(self, builder, eq):
        super(OhmicBCSpecies, self).initDiscreteEq(builder, eq)
        eq.semiconductor = builder.get(self.semiconductor)
        eq.semiconductor.alldone.depends(eq.evaluate)
        eq.evaluate.depends(eq.semiconductor.load)

    def value(self, ctx, eq):
        return getitem(ctx.varsOf(eq.semiconductor)[
                       'conc_Ef'][id(eq.owner_eq)], eq._getIdx())


class OhmicContact(Coupled):
    def __init__(self, poisson, semiconductor, name):
        parts = [OhmicBCPoisson(name, semiconductor, poisson)]
        for s in semiconductor.species:
            if isinstance(s, WithDOS):
                parts.append(OhmicBCSpecies(name, semiconductor, s))
        super(OhmicContact, self).__init__(parts)
        self.semiconductor = semiconductor
        self.poisson = poisson
        self.name = name
