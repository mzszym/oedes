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

# From https://www.nrel.gov/grid/solar-resource/spectra-am1.5.html

from oedes.optical.func import InterpolatedFunctionOfWavelength
import os
from six import BytesIO
import numpy as np
import zipfile

__all__ = ['AM0', 'AM1_5', 'AM1_5_global', 'AM1_5_direct']


def load():
    path = os.path.dirname(__file__)
    c = zipfile.ZipFile(os.path.join(path, 'astmg173.zip'))
    csv_data = c.read('ASTMG173.csv')
    data = np.genfromtxt(BytesIO(csv_data), delimiter=',', skip_header=2)
    data[:, 0] *= 1e-9
    data[:, 1:] *= 1e9
    AM0 = InterpolatedFunctionOfWavelength(data[:, 0], data[:, 1])
    AM1_5_global = InterpolatedFunctionOfWavelength(data[:, 0], data[:, 2])
    AM1_5_direct = InterpolatedFunctionOfWavelength(data[:, 0], data[:, 3])
    return AM0, AM1_5_global, AM1_5_direct


AM0, AM1_5_global, AM1_5_direct = load()
AM1_5 = AM1_5_global
