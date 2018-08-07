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

from oedes.optical.func import InterpolatedFunctionOfWavelength
import numpy as np

__all__ = ['GaussianSpectrum']


class GaussianSpectrum(InterpolatedFunctionOfWavelength):
    def __init__(self, total_power, center, sigma, cutoff=2., points=101):
        a = cutoff*sigma
        d = np.linspace(-a, a, points)
        y = np.exp(-d*d/(2.*sigma*sigma))
        y *= total_power / np.trapz(y, d)
        super(GaussianSpectrum, self).__init__(d+center, y)
