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

__all__ = ['Domain']
from .equations import ConstTemperature, PoissonEquation, RamoShockleyCurrentCalculation, ChargedSpecies
from .source import BulkSource
from .compose import ModelAdapter, Coupled


class Domain(ModelAdapter, Coupled):
    "Domain in space with common Poisson's equation and thermal model"

    def __init__(self, mesh=None, calculate_current=True):
        super(Domain, self).__init__()
        self.poisson = PoissonEquation(mesh=mesh)
        self.species = []
        self.sources = []
        self.ordering = 'cell'
        self.thermal = ConstTemperature()
        self.other = []
        self.calculate_current = calculate_current
        self._current_calculation = None

    def _check(self, check, value):
        if check is not None:
            assert check is value or not check
        return value

    @property
    def parts(self):
        yield self.thermal
        assert isinstance(self.poisson, PoissonEquation)
        yield self.poisson
        for eq in self.species:
            assert isinstance(eq, ChargedSpecies)
            eq.poisson = self._check(eq.poisson, self.poisson)
            eq.thermal = self._check(eq.thermal, self.thermal)
            yield eq
        for source in self.sources:
            assert isinstance(source, BulkSource)
            yield source
        for other in self.other:
            yield other
        if self.calculate_current:
            if self._current_calculation is None:
                self._current_calculation = RamoShockleyCurrentCalculation(
                    (self.poisson,))
            yield self._current_calculation
