# -*- coding: utf-8; -*-
#
# oedes - organic electronic device simulator
# Copyright (C) 2017 Marek Zdzislaw Szymanski (marek@marekszymanski.com)
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

from sparsegrad.forward import nvalue

def isscalar(x):
    return not forward.nvalue(x).shape


def getitem(x, idx):
    if idx is None:
        return x
    if not isscalar(idx) and not len(idx):
        return np.zeros(0,dtype=nvalue(x).dtype)
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

def branch(cond, iftrue, iffalse):
    if isscalar(cond):
        if cond:
            return iftrue(None)
        else:
            return iffalse(None)
    n=len(cond)
    r=np.arange(len(cond))
    ixtrue=r[cond]
    ixfalse=r[np.logical_not(cond)]
    vtrue = iftrue(ixtrue)
    vfalse = iffalse(ixfalse)
    if isscalar(vtrue):
        vtrue=np.ones_like(ixtrue)*vtrue
    if isscalar(vfalse):
        vfalse=np.ones_like(ixfalse)*vfalse
    return sparsesum_bare(n, [(ixtrue,vtrue), (ixfalse,vfalse)])
