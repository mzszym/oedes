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

# # Transport of charged species
#
# $v=z \mu E$
#
# $D=\mu V_t$
#
# $V_t=\frac{k_B T}{q}$
#
# Fluxes for electrons and holes are
#
# $j_n=-\mu_e (n E + V_t \nabla n)$
#
# $j_p=\mu_h (p E - V_t \nabla p)$
#
# Electric currents are
#
# $J_n=q \mu_e (n E + V_t \nabla n)$
#
# $J_p=q \mu_h (p E - V_t \nabla p)$

from .cell import FVMConservationEquation
from oedes.functions import ScharfetterGummelFlux
import scipy.constants
import numpy as np
from oedes import logs


class FVMTransportEquation(FVMConservationEquation):
    "Advection-diffusion equation discretized with Scharfetter-Gummel scheme"

    def faceflux(self, c, v, D, full_output=False):
        """
        Return Scharfetter-Gummel approximation of flux

        Parameters:
        -----------
        c: vector
            concentration per cell
        v: vector
            velocity per face
        D: vector
            diffusion term per face
        full_output: bool
            If True, returns dict (flux,cmid,cgrad,flux_v,flux_d), otherwise flux.

        Returns:
        --------
        flux:
            Flux per face
        cmid:
            Concetration per face
        cgrad:
            Concetration gradient per face
        flux_v:
            Advection term of flux per face
        flux_d:
            Diffusion term of flux per face
        """
        dr = self.mesh.faces['dr']
        cfrom = c[self.mesh.faces['-']]
        cto = c[self.mesh.faces['+']]
        if not full_output:
            return ScharfetterGummelFlux(
                dr, cfrom, cto, v, D, full_output=False)
        flux, cmid, cgrad = ScharfetterGummelFlux(
            dr, cfrom, cto, v, D, full_output=True)
        return dict(flux=flux, cmid=cmid, cgrad=cgrad,
                    flux_v=cmid * v, flux_D=-D * cgrad)

    def update(self, x, *args):
        # Do not allow negative values
        xe = np.take(x, self.idx)
        xnew = np.where(xe < 0., 0., xe)
        if np.any(xe != xnew):
            pass
            #logs.fvm.info('update(%r): clipping concentration c<0, min(c)=%r'%(self, np.amin(xe)))
        x[self.idx] = xnew

    def scaling(self, xscaling, fscaling):
        lunit = 1e-9
        tunit = 1e-12
        xscaling[self.idx] = lunit**-3
        fscaling[self.idx] = lunit * tunit


class FVMTransportChargedEquation(FVMTransportEquation):
    "Advection-diffusion equation for charged species"

    def __init__(self, mesh, name, z):
        super(FVMTransportChargedEquation, self).__init__(mesh, name)
        self.z = z
        self.ze = z * scipy.constants.elementary_charge

    def v(self, mu, E):
        return E * mu * self.z

    def D(self, mu, D_mu):
        return mu * D_mu

    def chargedensity(self, c):
        return c * self.ze

    def facecurrent(self, faceflux):
        return faceflux * self.ze
