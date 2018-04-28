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

from oedes import ad
import numpy as np
from . import gdos
from .physics import spacing_to_N0 as atoN0
from .physics import N0_to_spacing as N0toa

# In all functions, recommended 3<=nsigma<=6

g1_max_c = 0.1
g2_max_En = 2.
g2_max_En2 = g2_max_En**2
g3_max_c = 0.5


def g1(nsigma, c):
    """
    EGDM g1
    nsigma=sigma/kT: must be nsigma>=1.
    c must be 0<=c<=1, recommended c<0.1
    """
    delta = 2.0 * (ad.log(nsigma**2 - nsigma) -
                   ad.log(ad.log(4))) / (nsigma**2)
    return ad.exp(0.5 * (nsigma**2 - nsigma) * (2 * c)**delta)


def g2_En2(nsigma, En2):
    """
    EGDM g2(En2)
    nsigma=sigma/kT, recommended 3<=nsigma<=6
    En2=(E*a/sigma)**2, recommended En2<4.
    """
    return ad.exp(0.44 * (nsigma**(3. / 2.) - 2.2)
                  * (ad.sqrt(1 + 0.8 * En2) - 1))


def g2(nsigma, En):
    """EGDM g2(En), nsigma=sigma/kT, En=E*a/sigma"""
    return g2_En2(nsigma, En**2)


def mu0t(nsigma, mu0):
    """EGDM mu_0(T) - temperature dependent mobility prefactor in
    function of temperature independent prefactor"""
    return mu0 * 1.8e-9 * ad.exp(-0.42 * nsigma**2)


def mu0t_inverse(nsigma, mu0t):
    return mu0t / (1.8e-9 * ad.exp(-0.42 * nsigma**2))


def nsigma(sigma, Vt):
    """Return nsigma=sigma/kT, sigma is in eV, thermal voltage Vt=kT/q is in V"""
    return sigma / Vt


def nsigma_inverse(nsigma, Vt):
    return nsigma * Vt


def g3(nsigma, c, impl=gdos.defaultImpl, b=None):
    return gdos.diffusion_enhancement(nsigma * np.sqrt(2.), c, impl, b=b)
