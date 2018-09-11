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

__all__ = ['BandEnergy', 'BandEnergyFromParams', 'WithDOS', 'DOS']

from oedes.utils import SubEquation
from .charged import ChargedSpecies
from oedes.models import solver
from oedes.ad import getitem


class BandEnergy(SubEquation):
    "Abstract class for calculation of band energy"

    def energy(self, parent, ctx, parent_eq):
        raise NotImplementedError()


class BandEnergyFromParams(BandEnergy):
    "Band energy as `.energy`` parameter"

    def energy(self, parent, ctx, parent_eq):
        return ctx.param(parent_eq, 'energy')


class WithDOS(ChargedSpecies):
    """
    Charged species with DOS

    Outputs:
    --------
    Ef : cell variable
        quasi Fermi energy [ eV ]
    phi_f : cell variable
        quasi Fermi potential [V]
    Eband : cell variable
        band energy [ eV ]
    phi_band : cell variable
        band equivalent potential (Eband/-q)

    """

    def __init__(self, *args, **kwargs):
        dos = kwargs.pop('dos', None)
        band = kwargs.pop('band', BandEnergyFromParams())
        super(WithDOS, self).__init__(*args, **kwargs)
        self.dos = dos
        self.band = band

    def load_concentration(self, ctx, eq):
        super(WithDOS, self).load_concentration(ctx, eq)
        if isinstance(ctx.solver, solver.RamoShockleyCalculation):
            return
        if self.dos is None:  # TODO
            return
        potential = ctx.varsOf(eq.poisson)['potential']
        Ebandv = self.band.energy(self, ctx, eq)
        Eband = -potential + Ebandv
        ctx.varsOf(eq).update(Ebandv=Ebandv, Eband=Eband)
        Ef = self.dos.QuasiFermiLevel(ctx, eq)
        ctx.varsOf(eq).update(Ef=Ef)
        if ctx.wants_output:
            ctx.outputCell([eq, 'Ef'], Ef, unit=ctx.units.eV)
            ctx.outputCell([eq, 'Eband'], Eband, unit=ctx.units.eV)
            ctx.outputCell([eq, 'phi_f'], -Ef, unit=ctx.units.V)
            ctx.outputCell([eq, 'phi_band'], -Eband, unit=ctx.units.V)

    def buildDiscreteEq(self, builder, obj):
        super(WithDOS, self).buildDiscreteEq(builder, obj)
        obj.dos = self.dos
        obj.band = self.band

    def initDiscreteEq(self, builder, obj):
        super(WithDOS, self).initDiscreteEq(builder, obj)
        if self.dos is None:  # TODO
            return
        self.band.subInit(self, builder, obj)
        self.dos.subInit(self, builder, obj)


class DOS(SubEquation):
    def Ef(self, ctx, eq):
        "Return band-referenced Fermi level from concentration"
        raise NotImplementedError()

    def Ef_correction(self, ctx, eq):
        "Return Ef - kT log c"
        raise NotImplementedError()

    def c(self, ctx, eq, Ef, numeric_parameters):
        "Return concentration from band-referenced Fermi level"
        raise NotImplementedError()

    def QuasiFermiLevel(self, ctx, eq):
        assert eq.z in [1, -1]
        Eband = ctx.varsOf(eq)['Eband']
        return Eband - eq.z * self.Ef(ctx, eq)

    def _concentration(self, ctx, eq, idx, imref, ref, **kwargs):
        assert eq.z in [1, -1]
        if idx is not None:
            Eband = getitem(ctx.varsOf(eq)[ref], idx)
        else:
            Eband = ctx.varsOf(eq)[ref]
        return self.c(ctx, eq, -(imref - Eband) / eq.z, **kwargs)

    def concentration(self, ctx, eq, idx, imref, **kwargs):
        return self._concentration(ctx, eq, idx, imref, 'Eband', **kwargs)

    def concentrationv(self, ctx, eq, idx, imref, **kwargs):
        return self._concentration(ctx, eq, idx, imref, 'Ebandv', **kwargs)

    def D_mu(self, parent_eq, ctx, eq):
        "Return D/mu"
        raise NotImplementedError()
