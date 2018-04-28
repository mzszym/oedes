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

import scipy.constants
from oedes import ad
import numpy as np


def LangevinRecombination(mu_n, mu_p, n, p, epsilon, npi):
    "Langevin recombination rate"
    return (n * p - npi) * scipy.constants.elementary_charge / \
        epsilon * (mu_n + mu_p)


def EmtageODwyerBarrierLowering(F, epsilon):
    """
    Emtage-O'Dwyer barrier lowering formula
    F must be positive
    """
    k = scipy.constants.elementary_charge / (4 * np.pi * epsilon)
    return np.sqrt(F * k)


def ThermalVoltage(T):
    return scipy.constants.Boltzmann * T / scipy.constants.elementary_charge


def spacing_to_N0(a):
    return a**-3.


def N0_to_spacing(N0):
    return N0**-(1. / 3.)


def MottGurney(epsilon_r, mu, V, L):
    return 9. / 8. * scipy.constants.epsilon_0 * epsilon_r * mu * V**2 / L**3


ManyRakavy_t1 = 2. * (1. - np.exp(-0.5))


def ManyRakavy(mu, V, L):
    return ManyRakavy_t1 * L**2 / (mu * V)


def OnsagerFunction(b):
    """
    Onsager function of real variable J1(2 \sqrt(-2b))/sqrt(-2b), b>=0
    """
    def f(x):
        return scipy.special.i1(2. * x) / x

    def df(args, f):
        x, = args
        u = 2. * x
        yield lambda: 2. * (scipy.special.i0(u) - f) / x
    return ad.custom_function(f, df)(np.sqrt(2. * b + 1e-10))
