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
from oedes import functions


class DirichletBC(BoundaryEquation):

    def __init__(self, name):
        self.name = name

    def setUp(self, eq):
        self.bidx = eq.mesh.getBoundary(self.name)['bidx']
        BoundaryEquation.setUp(self, eq)

    def _residuals(self, eq, v, FdS, b_value):
        i = eq.mesh.boundary.idx[self.bidx]
        return v[i] - b_value


class AppliedVoltage(DirichletBC):

    def residuals(self, eq, v, FdS, vars):
        assert isinstance(eq, Poisson)
        params = vars['params']
        return self._residuals(eq, v, FdS, params[
                               '%s.voltage' % self.name] - params['%s.workfunction' % self.name])


class Zero(DirichletBC):

    def residuals(self, eq, v, FdS, vars):
        return self._residuals(eq, v, FdS, 0)


class DirichletFromParams(DirichletBC):

    def residuals(self, eq, v, FdS, vars):
        return self._residuals(eq, v, FdS, vars['params'][
                               '%s.%s' % (eq.prefix, self.name)])


class Internal(BoundaryEquation):
    eq_from = None
    bidx_from = None

    def setUp(self, eq):
        BoundaryEquation.setUp(self, eq)
        self.conservation_to = self.eq_from.idx[
            self.eq_from.mesh.boundary.idx[self.bidx_from]]


class Equal(Internal):

    def __init__(self, other_eq, name):
        self.name = name
        self.eq_from = other_eq

    def setUp(self, eq):
        if self.eq_from is None:
            self.eq_from = eq
        bto, bfrom = eq.mesh.getSharedBoundary(self.eq_from.mesh, self.name)
        self.bidx = bto['bidx']
        self.bidx_from = bfrom['bidx']
        Internal.setUp(self, eq)

    def residuals(self, eq, v, FdS, vars):
        x = vars['x']
        return x[eq.idx[eq.mesh.boundary.idx[self.bidx]]] - \
            x[self.eq_from.idx[self.eq_from.mesh.boundary.idx[self.bidx_from]]]


class FermiLevelEqual(Equal):

    def residuals(self, eq, v, FdS, vars):
        ixto = eq.mesh.boundary.idx[self.bidx]
        ixfrom = self.eq_from.mesh.boundary.idx[self.bidx_from]
        Effrom = vars['eqvars'][id(self.eq_from)]['Ef'][
            self.eq_from.prefix][ixfrom]
        cto = vars['eqvars'][id(eq)]['model'].species_dos[
            self.eq_from.prefix].concentration(eq, vars, ixto, Effrom)
        return vars['x'][eq.idx[ixto]] - cto


class FermiLevelEqualElectrode(DirichletBC):
    F_eps = 1e-10

    def __init__(self, name, image_force=False, **kwargs):
        super(FermiLevelEqualElectrode, self).__init__(name, **kwargs)
        self.image_force = image_force

    def image_correction(self, eq, vars, ixto):
        if not self.image_force:
            return 0.
        n = eq.mesh.boundaries[self.name]['normal']
        F = eq.z * eq.mesh.dotv(vars['Ecellv'][ixto], n)
        F = where(F > self.F_eps, F, self.F_eps)
        return -eq.z * \
            functions.EmtageODwyerBarrierLowering(
                F, getitem(vars['epsilon'], ixto))

    def residuals(self, eq, v, FdS, vars):
        dos = vars['model'].species_dos[eq.prefix]
        ixto = eq.mesh.boundary.idx[self.bidx]
        params = vars['params']
        Ef_electrode = -params['%s.voltage' %
                               self.name] + self.image_correction(eq, vars, ixto)
        c_electrode = dos.concentration(eq, vars, ixto, Ef_electrode)
        return vars['x'][eq.idx[ixto]] - c_electrode
