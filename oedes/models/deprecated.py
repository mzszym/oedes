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

__all__ = [
    'BaseModel',
    'std',
    'electronic_device',
    'add_transport',
    'add_ions',
    'electrolyte',
    'holeonly',
    'electrononly',
    'bulk_heterojunction',
    'bulk_heterojunction_params']

from .domain import Domain
from .equations import AdvectionDiffusion, WithDOS


class BaseModel(Domain):
    """
    Basic device model:
    Supports arbitrary number of species (transported or not) coupled to Poisson's equation, with any form of
    boundary conditions and drift and diffusion velocities
    """

    def __init__(self, *args, **kwargs):
        super(BaseModel, self).__init__(*args, **kwargs)
        self.species_v_D = dict()
        self.species_dos = dict()

    @property
    def parts(self):
        for eq in self.species:
            name = eq.name
            if isinstance(eq, WithDOS):
                if name in self.species_dos:
                    eq.dos = self._check(eq.dos, self.species_dos[name])
            else:
                assert name not in self.species_dos
            if isinstance(eq, AdvectionDiffusion) in self.species_v_D:
                eq.v_D = self._check(eq.v_D, self.species_v_D[name])
            else:
                assert name not in self.species_v_D
        return super(BaseModel, self).parts

    @property
    def equations(self):
        return self.parts


from . import std
from .std import *
