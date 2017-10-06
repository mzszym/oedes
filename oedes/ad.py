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

from sparsegrad import *
import numpy as np
import scipy.sparse

nvalue = forward.nvalue


def isscalar(x):
    return forward.isscalar(forward.nvalue(x))


def custom_function(f, df):
    return wrapped_func(func.ufunc(func=f, deriv=df))


def sparsesum(n, terms, compress=False, unique=False):
    """
    Sum sparse vectors, where each term is given as (index,values)
    Resulting length n should be known in advance
    If compress, return as a spearse vector idx,sum
    If unique, only one term can contribute to each result
    """
    # s=np.zeros(n)
    sdata = [np.zeros(0, dtype=np.double)]
    scol = [np.zeros(0, dtype=np.int)]
    data = [np.zeros(0, dtype=np.double)]
    row = [np.zeros(0, dtype=np.int)]
    col = [np.zeros(0, dtype=np.int)]
    M = None
    used = np.zeros(n, dtype=np.bool)
    for idx, f in terms:
        v = np.asarray(nvalue(f))
        # result_dtype=np.result_type(s.dtype,v.dtype)
        # if s.dtype is not result_dtype:
        #    s=np.asarray(s,dtype=s.dtype)
        sdata.append(v)
        scol.append(idx)
        # s[idx]+=v
        if unique and np.any(used[idx]):
            raise RuntimeError('duplicate values')
        used[idx] = True
        if isinstance(f, expr_base):
            M = f.M
            fx = f.gradient.tocsr().tocoo()
            gw = fx.shape[1]
            assert fx.shape[0] == len(idx)
            row.append(idx[fx.row])
            col.append(fx.col)
            data.append(fx.data)
    sdata = np.hstack(sdata)
    scol = np.hstack(scol)
    s = np.asarray(scipy.sparse.csr_matrix(
        (sdata, (np.zeros_like(scol), scol)), shape=(1, n)).todense()).flatten()
    if M is None:
        result = s
    else:
        data = np.hstack(data)
        col = np.hstack(col)
        row = np.hstack(row)
        d = scipy.sparse.csr_matrix((data, (row, col)), shape=(s.shape[0], gw))
        result = forward.value(value=s, dvalue=d, M=M)
    if compress:
        return np.arange(n)[used], result[used]
    else:
        return result


def getitem(x, idx):
    if isscalar(x):
        return x
    else:
        return x[idx]
