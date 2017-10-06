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

# # Poisson's equation
#
# $\nabla \cdot (\epsilon \nabla \varphi) = \rho$

from .cell import CellEquation


class Poisson(CellEquation):
    "Poisson equation"

    def __init__(self, mesh, name='poisson', **kwargs):
        CellEquation.__init__(self, mesh, name=name, **kwargs)

    def E(self, phi):
        "Return magnitude of electric field at faces"
        return -self.mesh.facegrad(phi)

    def displacement(self, E, epsilon):
        return E * epsilon

    faceflux = displacement

    @property
    def transientvar(self):
        return 0.
