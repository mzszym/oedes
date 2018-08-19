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

import warnings
from six import BytesIO, PY2
from oedes.optical.func import FunctionOfWavelength, makeComplexFunctionOfWavelength, InterpolatedFunctionOfWavelength, ConstFunctionOfWavelength
import numpy as np
import yaml
from oedes.optical.func import Material

__all__ = ['RefractiveIndexInfoMaterial']


class RefractiveIndexInfoMaterialWarning(UserWarning):
    pass


class Formula(FunctionOfWavelength):
    def __init__(self, limits, coefficients, min_coeff=1, odd_number=1):
        if len(coefficients) < min_coeff or (
                odd_number is not None and len(coefficients) % 2 != odd_number):
            raise ValueError(
                '{name}: wrong number of coefficients'.format(
                    name=self.__class__.__name__))
        self.coefficients = coefficients
        self.limits = limits
        self.undefined = False

    def __call__(self, wavelength):
        return self.calculate(wavelength*1e6)


class Sellmeier(Formula):
    def __init__(self, limits, coefficients, variant_2=False):
        self.variant_2 = variant_2
        super(Sellmeier, self).__init__(limits, coefficients)

    def calculate(self, wavelength):
        w2 = wavelength*wavelength
        s = 1 + self.coefficients[0]
        for i in range(1, len(self.coefficients), 2):
            u = self.coefficients[i+1]
            if not self.variant_2:
                u = u*u
            s = s + self.coefficients[i]*w2 / (w2-u)
        return np.sqrt(s)


class Sellmeier_2(Sellmeier):
    def __init__(self, limits, coefficients):
        super(Sellmeier_2, self).__init__(limits, coefficients, variant_2=True)


class Cauchy(Formula):
    def calculate(self, wavelength):
        w = wavelength
        s = self.coefficients[0]
        for i in range(1, len(self.coefficients), 2):
            s = s + self.coefficients[i]*w**self.coefficients[i+1]
        return s


class Polynominal(Cauchy):
    def calculate(self, wavelength):
        return np.sqrt(super(Polynominal, self).calculate(wavelength))


class RefractiveIndexInfo(Formula):
    def __init__(self, *args):
        super(RefractiveIndexInfo, self).__init__(*args, min_coeff=9)

    def calculate(self, wavelength):
        w = wavelength
        w2 = w*w
        c = self.coefficients
        s = c[0]+(c[1]*w**c[2]/(w2-c[3]**c[4]))+(c[5]*w**c[6]/(w2-c[7]**c[8]))
        for i in range(9, len(self.coefficients), 2):
            s = s + self.coefficients[i]*w**self.coefficients[i+1]
        return np.sqrt(s)


class Gases(Formula):
    def calculate(self, wavelength):
        w = wavelength
        iw2 = 1./(w*w)
        s = 1 + self.coefficients[0]
        for i in range(1, len(self.coefficients), 2):
            s = s + self.coefficients[i]/(self.coefficients[i+1]-iw2)
        return s


class Herzberger(Formula):
    def __init__(self, limits, coefficients):
        super(
            Herzberger,
            self).__init__(
            limits,
            coefficients,
            min_coeff=4,
            odd_number=None)

    def calculate(self, wavelength):
        w2 = wavelength*wavelength
        c = self.coefficients
        u = 1./(w2-0.028)
        s = c[0]+u*(c[1]+c[2]*u)
        v = w2
        for i in range(3, len(self.coefficients)):
            s = s + c[i]*v
            v = v*v
        return s


class Retro(Formula):
    def __init__(self, limits, coefficients):
        if len(coefficients) != 4:
            raise ValueError('expecting 4 coefficients')
        super(Retro, self).__init__(limits, coefficients, odd_number=0)

    def calculate(self, wavelength):
        c = self.coefficients
        w2 = wavelength*wavelength
        s = c[0]+c[1]*w2/(w2-c[2])+c[3]*w2
        return np.sqrt((2*s+1)/(1-s))


class Exotic(Formula):
    def __init__(self, limits, coefficients):
        if len(coefficients) != 6:
            raise ValueError('expecting 6 coefficients')
        super(Exotic, self).__init__(limits, coefficients, odd_number=0)

    def calculate(self, wavelength):
        c = self.coefficients
        w = wavelength
        u = (w-c[4])
        s = c[0]+c[1]/(w*w-c[2])+(c[3]*u/(u*u+c[5]))
        return np.sqrt(s)


def gen_from_str(s):
    if not PY2 and isinstance(s, str):
        s = bytes(s, 'utf-8')
    return np.genfromtxt(BytesIO(s))


class RefractiveIndexInfoMaterial(Material):
    def _warn(self, message):
        if not self.ignore_warnings:
            warnings.warn(message, RefractiveIndexInfoMaterialWarning)

    def __init__(self, stream, name=None, ignore_warnings=False):
        self.ignore_warnings = ignore_warnings
        if isinstance(stream, str):
            with open(stream, 'rb') as f:
                yml = yaml.safe_load(f)
        else:
            yml = yaml.safe_load(stream)

        self.references = yml.get('REREFERENCES', None)
        self.comments = yml.get('COMMENTS', None)
        self.specs = yml.get('SPECS', None)

        self.n_data = None
        self.k_data = None

        for data in yml['DATA']:
            data_type = data['type'].split()
            if data_type[0] == 'formula':
                f = self._process_formula
            elif data_type[0] == 'tabulated':
                f = self._process_tabulated
            else:
                self._warn(
                    'Skipping not understood data type {data_type}'.format(
                        data_type=data['type']))
            f(data, data_type[1:])

        if self.n_data is None:
            self._warn('n data not defined')
        if self.k_data is None:
            self._warn('k data not defined')
        if self.n_data is not None and self.k_data is not None:
            self.data = makeComplexFunctionOfWavelength(
                self.n_data, self.k_data)
        else:
            self.data = None

        super(RefractiveIndexInfoMaterial, self).__init__(refractive_index=self.data, name=name)

    def _process_tabulated(self, data, args):
        rows = gen_from_str(data['data'])
        if rows.ndim == 0 or not all(rows.shape):
            self._warn('skipping empty tabulated data')
            return
        if rows.ndim == 1:
            rows = rows[np.newaxis, ...]
        rows = np.atleast_2d(rows)
        rows = rows[np.argsort(rows[:, 0])]
        delta = np.diff(rows[:, 0])
        duplicate = delta == 0.
        if np.any(duplicate):
            ix = np.ones(len(rows), dtype=np.bool)
            ix[1:][duplicate] = False
            self._warn(
                'Removing duplicate wavelength values : {dup}'.format(dup=rows[~ix, 0]))
            rows = rows[ix]
        types = {'n': ((1, None), 2), 'k': ((None, 1), 2), 'nk': ((1, 2), 3)}
        if len(args) != 1:
            raise ValueError(
                'tabulated arguments not understood {args}'.format(
                    args=args))
        type_name, = args
        if type_name not in types:
            raise ValueError(
                'tabulated type not understood {type_name}'.format(
                    type_name=type_name))
        (n_source, k_source), ncols_required = types[type_name]
        nrows, ncols = rows.shape
        if ncols != ncols_required:
            raise ValueError(
                'invalid number of columns for type {type_name} : expected {ncols_required}, got {ncols}'.format(
                    type_name=type_name, ncols_required=ncols_required, ncols=ncols))
        wavelength = rows[:, 0]*1e-6

        def to_function(x, y):
            if len(x) > 1:
                return InterpolatedFunctionOfWavelength(x, y)
            else:
                return ConstFunctionOfWavelength(y[0], (x[0], x[0]))
        if n_source is not None:
            if self.n_data is not None:
                self._warn('tabulated data redefines n data')
            self.n_data = to_function(wavelength, rows[:, n_source])
        if k_source is not None:
            if self.k_data is not None:
                self._warn('tabulated data redefines k data')
            self.k_data = to_function(wavelength, rows[:, k_source])

    def _process_formula(self, data, args):
        if self.n_data is not None:
            self._warn('formula is redefining n data')
        if len(args) != 1:
            raise ValueError(
                'formula arguments not understood {args}'.format(args))
        formula_type, = args
        formula_types = {
            '1': Sellmeier,
            '2': Sellmeier_2,
            '3': Polynominal,
            '4': RefractiveIndexInfo,
            '5': Cauchy,
            '6': Gases,
            '7': Herzberger,
            '8': Retro,
            '9': Exotic
        }
        if formula_type not in formula_types:
            raise ValueError(
                'unknown formula type {formula_type}'.format(
                    formula_type=formula_type))
        coefficients = gen_from_str(data['coefficients'])
        limits = None
        if 'range' in data:
            limits = data['range']
        if 'wavelength_range' in data:
            if limits is not None:
                raise ValueError(
                    'cannot specify range and wavelength_range at once')
            limits = data['wavelength_range']
        limits = gen_from_str(limits)*1e-6
        if limits.shape != (2,) or limits[0] > limits[1]:
            raise ValueError(
                'wavelength limits not understood {limits}'.format(
                    limits=limits))
        if self.n_data is not None:
            raise ValueError('redefining n')
        self.n_data = formula_types[formula_type](limits, coefficients)
