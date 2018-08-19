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

import scipy.interpolate
import numpy as np

__all__ = [
    'FunctionOfWavelength',
    'ConstFunctionOfWavelength',
    'InterpolatedFunctionOfWavelength',
    'wavelength_range',
    'makeComplexFunctionOfWavelength',
    'Material']


class FunctionOfWavelength(object):
    "This defines __call__(wavelength) and limits, with wavelength in m"
    pass


class InterpolatedFunctionOfWavelength(FunctionOfWavelength):
    def __init__(self, x, y):
        if np.any(np.diff(x) <= 0):
            raise ValueError('x values not sorted : {x}'.format(x=x))
        self.interp = scipy.interpolate.interp1d(
            x, y, kind='linear', bounds_error='True', assume_sorted=True)
        self.xdata = x
        self.ydata = y
        self.limits = (np.amin(x), np.amax(x))

    def __call__(self, wavelength):
        return self.interp(wavelength)


class ConstFunctionOfWavelength(FunctionOfWavelength):
    def __init__(self, value, limits=(-np.inf, np.inf)):
        self.value = value
        self.limits = np.asarray(limits)

    def __call__(self, wavelength):
        return self.value*np.ones_like(wavelength)


def wavelength_range(*functions):
    return tuple([max(u.limits[0] for u in functions),
                  min(u.limits[1] for u in functions)])


class makeComplexFunctionOfWavelength(object):
    def __init__(self, real_part, imag_part):
        self.limits = wavelength_range(real_part, imag_part)
        self.real_part = real_part
        self.imag_part = imag_part

    def __call__(self, wavelength):
        return self.real_part(wavelength) + self.imag_part(wavelength)*1.J


class Material(object):
    def __init__(self, refractive_index, name=None):
        self.refractive_index = refractive_index
        self.name = name
