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

import numpy as np
from sparsegrad import *
from sparsegrad.base import *
from sparsegrad import forward
from sparsegrad.forward import nvalue
from sparsegrad.sparsevec import *


def isscalar(x):
    return not forward.nvalue(x).shape


def getitem(x, idx):
    if idx is None:
        return x
    if not isscalar(idx) and not len(idx):
        return np.zeros(0, dtype=nvalue(x).dtype)
    if isscalar(x):
        return x
    else:
        return x[idx]


sparsesum = sparsesum_bare


def custom_function(f, df):
    f = expr.wrapped_func(func.custom_func(f, df))

    def _(*args):
        return f(*args)
    return f
