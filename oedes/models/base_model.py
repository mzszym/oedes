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

#from oedes.ad import *
from .equations import ConstTemperature, Poisson, NewCurrentCalculation, Transport
from .source import BulkSource
from .compose import ModelAdapter, Coupled
import itertools
import weakref


class BaseModel(ModelAdapter, Coupled):
    """
    Basic device model:
    Supports arbitrary number of species (transported or not) coupled to Poisson's equation, with any form of
    boundary conditions and drift and diffusion velocities
    """

    def __init__(self):
        super(BaseModel, self).__init__()
        self.poisson = None
        self.species = []
        self.sources = []
        self.species_v_D = dict()
        self.species_dos = dict()
        self.additional_charge = []
        self.ordering = 'cell'
        self._current_calculation = None
        self.thermal = ConstTemperature()

    @property
    def parts(self):
        def check(check, value):
            if check is not None:
                assert check is value or not check
            return value

        yield self.thermal
        assert isinstance(self.poisson, Poisson)
        self.poisson.additional_charge = check(
            self.poisson.additional_charge, self.additional_charge)
        yield self.poisson
        for eq in self.species:
            assert isinstance(eq, Transport)
            eq.poisson = check(eq.poisson, self.poisson)
            eq.thermal = check(eq.thermal, self.thermal)
            if eq.prefix in self.species_dos:
                eq.dos = check(eq.dos, self.species_dos[eq.prefix])
            else:
                eq.dos = check(eq.dos, None)
            eq.v_D = check(eq.v_D, self.species_v_D[eq.prefix])
            yield eq
        for eq in [self.poisson] + self.species:
            for bc in eq.bc:
                bc._owner_eq_weak = check(bc._owner_eq_weak, weakref.ref(eq))
                if bc.convergenceTest is None:
                    bc.convergenceTest = eq.defaultBCConvergenceTest
                yield bc
        for source in self.sources:
            assert isinstance(source, BulkSource)
            yield source
        if self._current_calculation is None:
            self._current_calculation = NewCurrentCalculation((self.poisson,))
        yield self._current_calculation

    @property
    def equations(self):
        return self.parts
