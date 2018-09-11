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

from oedes.models import BaseModel, electronic_device
from oedes.fvm import mesh1d
from oedes import solve
import numpy as np
from numpy.testing import assert_allclose


def make_simple_device(voltage=0., polarity='p',
                       x=np.linspace(0., 100e-9, 10)):
    model = BaseModel()
    mesh = mesh1d(x)
    electronic_device(model, mesh, polarity)
    params = {'T': 300., 'electrode0.voltage': voltage, 'electrode0.workfunction': 0, 'electrode1.voltage': 0, 'electrode1.workfunction': 0,
              'hole.N0': 1e27, 'electron.N0': 1e27, 'hole.mu': 1e-9, 'electron.mu': 1e-9, 'electron.energy': 0, 'hole.energy': 0, 'epsilon_r': 3.}
    model.setUp()
    return model, mesh, params


def check(model, params, v, p, res1, res2, dtype=np.double):
    x = np.array(model.X, dtype=dtype)
    x[model.poisson.idx] = v
    x[model.species[0].idx] = p

    def residuals(x):
        return model.residuals(0., x, 0. * x, params)

    def resnorm(x):
        return np.linalg.norm(residuals(x))
    assert resnorm(x) < res1
    xr = solve(model, x, params, maxiter=3)
    assert resnorm(xr) < res2


def test_0v():
    v = np.array([0., 0.21687333, 0.26445226, 0.28515601, 0.29379431,
                  0.29379431, 0.28515601, 0.26445226, 0.21687333, 0.])
    p = np.array([1.00000000e+27, 2.27345935e+23, 3.60907604e+22,
                  1.62027335e+22, 1.16003929e+22, 1.16003929e+22,
                  1.62027335e+22, 3.60907604e+22, 2.27345935e+23,
                  1.00000000e+27])
    model, _, params = make_simple_device()
    check(model, params, v, p, 2e24, 1e16)
    check(model, params, v, p, 2e24, 1e16, dtype=np.longdouble)


def test_1v():
    v = np.array([1., 1.20799767, 1.20085886, 1.12804101, 1.01286933,
                  0.86470611, 0.68875172, 0.48838939, 0.26602888, 0.])
    p = np.array([1.00000000e+27, 2.88907411e+23, 8.82005642e+22,
                  5.68770876e+22, 4.43044288e+22, 3.73208501e+22,
                  3.27774890e+22, 2.95414125e+22, 5.86423921e+22,
                  1.00000000e+27])
    model, _, params = make_simple_device(1.)
    check(model, params, v, p, 2e24, 1e18)
