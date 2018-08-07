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

__all__ = ['Electroneutrality']

import numpy as np
import scipy.optimize
from oedes.utils import Calculation
from oedes.models import solver
from oedes.ad import exp, log, sqrt, nvalue, custom_function, forward, isscalar
from .dos import WithDOS
from .boltzmann import BoltzmannDOS
from oedes.functions import brent


def _same(seq):
    value = next(seq)
    if all([x is value for x in seq]):
        return value
    else:
        return None


class Electroneutrality(Calculation):
    "This finds Fermi level corresponding to electrical neutrality, as well as neutral charge concentrations"

    etol = 1e-15
    force_general = False

    def __init__(self, species, name=None):
        super(Electroneutrality, self).__init__(name)
        self.species = species
        with_dos = []
        other = []
        for s in self.species:
            if isinstance(s, WithDOS):
                with_dos.append(s)
            else:
                other.append(s)
        self.with_dos = with_dos
        self.other = other
        self.electron = [eq for eq in with_dos if eq.z == -1]
        self.hole = [eq for eq in with_dos if eq.z == 1]

    def initDiscreteEq(self, builder, eq):
        super(Electroneutrality, self).initDiscreteEq(builder, eq)
        with_dos = list(map(builder.get, self.with_dos))
        other = list(map(builder.get, self.other))
        species = with_dos + other
        eq.with_dos, eq.other, eq.species = with_dos, other, species
        eq.electron = list(map(builder.get, self.electron))
        eq.hole = list(map(builder.get, self.hole))
        for s in species:
            s.alldone.depends(eq.load)

        # check if same mesh and same poisson equation
        eq.mesh = _same(s.mesh for s in species)
        eq.poisson = _same(s.poisson for s in species)
        eq.load.depends(eq.poisson.load)
        for s in with_dos:
            eq.load.depends(s.thermal.load)

        # Check conditions for application of analytical solution:
        # - bandgap, two species, different charges +-e
        # - both use Boltzann DOS
        # - both use same effective temperature
        same_thermal = _same(s.thermal for s in with_dos)
        boltzmann = _same(isinstance(s.dos, BoltzmannDOS) for s in with_dos)
        bandgap = (self.electron and eq.hole and len(
            with_dos) == 2) or len(with_dos) == 1
        eq.analytical = same_thermal and boltzmann and bandgap
        if eq.analytical:
            eq.mesh = eq.with_dos[0].mesh
            eq.thermal = eq.with_dos[0].thermal

    def solve_analytical(self, ctx, eq, fixed):
        hole, = eq.hole
        electron, = eq.electron
        C = fixed
        Nc = electron.dos.N0(ctx, electron)
        Nv = hole.dos.N0(ctx, hole)
        Vt = ctx.varsOf(eq.thermal)['Vt']
        Ec = ctx.varsOf(electron)['Ebandv']
        Ev = ctx.varsOf(hole)['Ebandv']
        uc = exp(Ec/Vt)
        uv = exp(Ev/Vt)
        # TODO: rewrite to avoid overflow, reference to middle of bandgap
        Ef = Vt*log((C*uc + sqrt((C*C*uc + 4*Nc*Nv*uv)*uc))/(2*Nc))
        n = Nc*exp((Ef-Ec)/Vt)
        p = Nv*exp((Ev-Ef)/Vt)
        return Ef, dict({id(hole): p, id(electron): n})

    def solve_general(self, ctx, eq, fixed):
        def conc(Ef, numeric_parameters):
            for s in eq.with_dos:
                yield s, s.dos.concentrationv(ctx, s, None, Ef, numeric_parameters=numeric_parameters)
        nfixed = nvalue(fixed)

        def f(Ef, numeric_parameters=True):
            if numeric_parameters:
                charge = nfixed
            else:
                charge = fixed
            for eq, c in conc(Ef, numeric_parameters):
                charge = charge + c*eq.z
            return charge
        bandv = np.asarray([nvalue(ctx.varsOf(s)['Ebandv'])
                            for s in eq.with_dos])
        a = np.amax(bandv)
        b = np.amin(bandv)
        Ef_value = brent(f, a, b, xtol=self.etol, maxiter=50)
        g = f(Ef_value, False)
        if isinstance(g, forward.value):
            raise NotImplementedError('not tested')
            dg = f(forward.seed(Ef_value), True).deriv
            if not isscalar(dg):
                dg = dg.tocsr().diagonal()
            # must create value : (Ef_value, 1/dg*g')
            Ef = custom_function(lambda *args: Ef_value, lambda *args: dg)(g)
        else:
            Ef = Ef_value
        info = dict((id(eq), c) for eq, c in conc(Ef, False))
        return Ef, info

    def load(self, ctx, eq):
        super(Electroneutrality, self).load(ctx, eq)
        if isinstance(ctx.solver, solver.RamoShockleyCalculation):
            return
        fixed = 0
        for s in eq.other:
            fixed = fixed + nvalue(ctx.varsOf(s)['c']) * s.z
        if eq.analytical and not self.force_general:
            Efv, c = self.solve_analytical(ctx, eq, fixed)
        else:
            Efv, c = self.solve_general(ctx, eq, fixed)
        ctx.varsOf(eq)['Efv'] = Efv
        ctx.varsOf(eq)['conc_Ef'] = c
        Ef = -ctx.varsOf(eq.poisson)['potential'] + Efv
        ctx.outputCell([eq, 'Ef'], Ef, unit=ctx.units.eV)
        ctx.outputCell([eq, 'phi'], -Ef, unit=ctx.units.V)
        for s, sc in zip(eq.with_dos, self.with_dos):
            ctx.outputCell([eq, sc.name, 'c'], c[id(s)],
                           unit=ctx.units.concentration)
