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

# This tests if Poisson's equation is solved correctly in most simple cases

from oedes import *
from oedes.fvm import mesh1d
from oedes.models import BaseModel, Poisson, AppliedVoltage
L = 100e-9
v0 = 1.
v1 = -1.
mesh = mesh1d(L)


def run_poisson(bc):
    b = BaseModel()
    b.poisson = Poisson(mesh)
    b.poisson.bc = bc
    params = {'T': 300., 'electrode0.voltage': v0, 'electrode1.voltage': v1,
              'electrode0.workfunction': 0., 'electrode1.workfunction': 0., 'epsilon_r': 3.}
    b.setUp()
    solution = solve(b, b.X, params)
    out = b.output(0., solution, 0. * b.X, params)
    return out


def test_poisson_DirichletDirichlet():
    out = run_poisson([AppliedVoltage('electrode0'),
                       AppliedVoltage('electrode1')])
    assert np.allclose(out['E'], (v0 - v1) / L)
    assert np.allclose(out['potential'], v0 + (v1 - v0)
                       * mesh.cells['center'] / L)


def test_poisson_DirichletOpen():
    out = run_poisson([AppliedVoltage('electrode0')])
    assert np.allclose(out['E'], 0.)
    assert np.allclose(out['potential'], v0)


def test_poisson_OpenDirichlet():
    # # Open-Dirichlet
    out = run_poisson([AppliedVoltage('electrode1')])
    assert np.allclose(out['E'], 0.)
    assert np.allclose(out['potential'], v1)
