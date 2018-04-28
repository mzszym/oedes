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

import tmm
import scipy.interpolate
import numpy as np


class SimpleLayer:
    def __init__(self, thickness, n):
        self.d = thickness
        self._n = n

    def n(self, wavelength):
        if callable(self._n):
            return self._n(wavelength)
        else:
            return self._n


class SimpleAbsorption:
    def __init__(self, mesh, n, layers_before, layers_after, wavelength):
        wrap = SimpleLayer(mesh.length, n)
        layers = layers_before + [wrap] + layers_after
        d_list = ['inf'] + [u.d for u in layers] + ['inf']
        n_list = [1.] + [u.n(wavelength) for u in layers] + [1.]
        out = tmm.coh_tmm(
            pol='s',
            n_list=n_list,
            d_list=d_list,
            th_0=0.,
            lam_vac=wavelength)
        i = len(layers_before) + 1
        x = mesh.cells['center']
        absor = np.asarray(
            [tmm.position_resolved(i, x_, out)['absor'] for x_ in x])
        self.normalization = np.amax(absor)
        self.a = scipy.interpolate.interp1d(x, absor)

    def normalized(self, x):
        return self.a(x) / self.normalization
