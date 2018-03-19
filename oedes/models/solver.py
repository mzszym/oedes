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

class SolverObject(object):
    def __init__(self):
        pass

    @property
    def poissonOnly(self):
        return False


class PoissonOnly(SolverObject):
    @property
    def poissonOnly(self):
        return True


class RamoShockleyCalculation(PoissonOnly):
    def __init__(self, boundary_name, store):
        super(RamoShockleyCalculation, self).__init__()
        self.boundary_name = boundary_name
        self.store = store
