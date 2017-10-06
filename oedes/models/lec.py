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

from . import base
import scipy.constants
from oedes import bdf1adapt as _bdf1adapt
from oedes import solve as _solve
from oedes import SolverError
from . import std
import numpy as np
import sys


def LEC(mesh, zc=1, za=-1, **kwargs):
    "Return ready-to-evaluate LEC model"
    model = base.BaseModel()
    std.electronic_device(model, mesh, 'pn', **kwargs)
    std.add_ions(model, mesh, zc=zc, za=za)
    model.setUp()
    return model


def _find(model):
    eq = dict([(eq.prefix, eq) for eq in model.species])
    return eq['cation'], eq['anion']


def initial_salt(model, cinit, nc=1., na=1.):
    "Return initial state of device corresponding to uniformly distributed ions with salt concentration cinit"

    cation, anion = _find(model)
    assert nc > 0 and na > 0 and cation.z > 0 and anion.z < 0
    assert cation.z * nc == -anion.z * na
    x = np.zeros_like(model.X)
    x[cation.idx] = cinit * nc
    x[anion.idx] = cinit * na
    return x


def debye_length(c, epsi_r=3., T=300.):
    return np.sqrt(scipy.constants.epsilon_0 * epsi_r * scipy.constants.Boltzmann *
                   T / (2 * scipy.constants.elementary_charge**2 * c))


def total_ions(model, eq, x):
    assert not eq.bc, "Calculation valid only for no-flux boundary conditions"
    return np.sum((eq.mesh.cells['volume'] * x[eq.idx]))


class checker:
    "Special variant of solve which checks if ion concentration is conservation to required precision"

    # This may but should not be useful
    # Solver may not conserve total number of ions if Jacobian is almost
    # singular

    def __init__(self, model, xinit, rtol, atol):
        self.model = model
        self.cation, self.anion = _find(model)
        self.rtol = rtol
        self.atol = atol
        self.c_tot = total_ions(model, self.cation, xinit)
        self.a_tot = total_ions(model, self.anion, xinit)

    def solve(self, model, *args, **kwargs):
        assert model is self.model
        x = _solve(model, *args, **kwargs)
        if not np.allclose(total_ions(model, self.cation, x),
                           self.c_tot, atol=self.atol, rtol=self.rtol):
            raise SolverError()
        if not np.allclose(total_ions(model, self.anion, x),
                           self.a_tot, atol=self.atol, rtol=self.rtol):
            raise SolverError()
        return x


def bdf1adapt(model, x, params, t, t1, dt,
              ion_atol=0., ion_rtol=1e-6, **kwargs):
    "bdf1adapt which includes checking of ion conservation (which should be totally unnecessary)"

    solve_and_check = checker(model, x, rtol=ion_rtol, atol=ion_atol)
    return _bdf1adapt(model, x, params, t, t1, dt,
                      solve=solve_and_check.solve, **kwargs)


class monitor_species:
    "Simple monitor, which plots species in Jupyter"

    def __init__(self, model, plt, ylim_species=[1e10, 1e30]):
        self.model = model
        self.plt = plt
        self.ylim_species = ylim_species

    def __call__(self, t, x, xt, out):
        plt = self.plt
        for eq in self.model.species:
            plt.plot(eq.mesh.cells['center'] * 1e9,
                     out[eq.prefix + '.c'], label=eq.prefix)
        plt.yscale('log')
        plt.xlabel('Position [nm]')
        plt.ylabel('Concentration [$m^-3$]')
        plt.legend(loc=0)
        plt.title(str(t))
        plt.ylim(self.ylim_species)
        plt.show()


def monitor_time(t, x, xt, out):
    sys.stdout.write('\r%s' % t + ' ' * 20)


def equilibrium(model, params, cinit, t1, dt, monitor=lambda t, x,
                xt, out: None, nc=1., na=1., ion_atol=0., ion_rtol=1e-6, **kwargs):
    # Do not modify params, so create copy
    params = params.copy()
    params['electrode1.voltage'] = 0.
    params['electrode0.voltage'] = 0.
    xinit = initial_salt(model, cinit, nc=nc, na=na)
    for t, x, xt, outf in bdf1adapt(
            model, xinit, params, 0., t1, dt, ion_atol=ion_atol, ion_rtol=ion_rtol):
        monitor(t, x, xt, outf())
    return x


def species_v_D_limited_from_params(eq, vars):
    v, D = species_v_D_charged_from_params(eq, vars)
    return v * (1. - eq.mesh.faceaverage(vars['c'][eq.prefix + '.N0'])), D
