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

from .base import *
from .simple import *
from .boundary import *


class species_from_dos(object):
    # This is currently badly designed and will be refactored later

    def __init__(self, dos_class):
        self.dos_class = dos_class

    def _make_dos(self, eq, vars):
        params = vars['params']
        base = self.dos_class(params[eq.prefix + '.N0'], params['T'])
        return MO(base, params[eq.prefix + '.level'], eq.z)

    def v_D(self, eq, vars):
        v, D0 = species_v_D_charged_from_params(eq, vars)
        dos = self._make_dos(eq, vars)
        return v, eq.mesh.faceaverage(
            vars['mu'][eq.prefix] * dos.g(vars['c'][eq.prefix]))

    def bc(self, eq, facefluxes, vars):
        dos = self._make_dos(eq, vars)
        return bc_dirichlet(eq, vars['c'][eq.prefix], lambda u: dos.concentration(
            vars['params']['%s.workfunction' % u]))


class DOS(object):

    def Ef(self, eq, vars):
        "Return band-referenced Fermi level from concentration"
        raise NotImplementedError()

    def c(self, eq, Ef, vars):
        "Return concentration from band-referenced Fermi level"
        raise NotImplementedError()

    def D(self, eq, c, Ef, vars):
        "Return diffusion coefficient"
        raise NotImplementedError()

    def QuasiFermiLevel(self, eq, vars):
        assert eq.z in [1, -1]
        Eband = -vars['potential'] - vars['params'][eq.prefix + '.level']
        return Eband - eq.z * self.Ef(eq, vars)

    def concentration(self, eq, vars, idx, imref):
        assert eq.z in [1, -1]
        Eband = -vars['potential'][idx] - vars['params'][eq.prefix + '.level']
        return self.c(eq, -(imref - Eband) / eq.z, vars)


class BoltzmannDOS(DOS):
    c_eps = 1e-30
    c_limit = True
    Ef_max = None

    def Ef(self, eq, vars):
        c = vars['c'][eq.prefix]
        c = where(c > self.c_eps, c, self.c_eps)  # avoid NaN
        N0 = vars['params'][eq.prefix + '.N0']
        # c=where(c<N0,c,N0)
        return vars['Vt'] * np.log(c / N0)

    def c(self, eq, Ef, vars):
        if self.Ef_max is not None:
            Ef = where(Ef < self.Ef_max, Ef, self.Ef_max)
        N0 = vars['params'][eq.prefix + '.N0']
        v = Ef / vars['Vt']
        if self.c_limit:
            v = where(v < 0., v, 0.)
        c = np.exp(v) * N0
        return c

    def D(self, eq, c, Ef, vars):
        return vars['Vt']

    def v_D(self, eq, vars):
        return species_v_D_charged_from_params(eq, vars)
