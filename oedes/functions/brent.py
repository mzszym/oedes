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

__all__ = ['brent']

import numpy as np


def sort(a, b, fa, fb):
    "if |f(a)|<|f(b)| swap(a,b)"
    swap = np.abs(fa) < np.abs(fb)

    def aswap(s, a, b):
        a_ = np.where(s, b, a)
        b_ = np.where(s, a, b)
        return a_, b_
    a, b = aswap(swap, a, b)
    fa, fb = aswap(swap, fa, fb)
    return a, b, fa, fb


def converged(a, b, s, fa, fb, fs, xtol):
    cb = fb == 0.
    if s is not None:
        cs = fs == 0.
    else:
        cs = False
    ctol = abs(a-b) < 2.*xtol
    if not np.all(np.logical_or(np.logical_or(cb, cs), ctol)):
        return None
    x = (a+b)/2
    x = np.where(cb, b, x)
    if s is not None:
        x = np.where(cs, s, x)
    return x


def quadratic(a, b, c, fa, fb, fc):
    with np.errstate(divide='ignore', invalid='ignore'):
        dab = fa-fb
        dbc = fb-fc
        dca = fc-fa
        x = -a*fb*fc/(dab*dca)-b*fa*fc/(dab*dbc)-c*fa*fb/(dca*dbc)
        return x


def secant(a, b, fa, fb):
    return b-fb*(b-a)/(fb-fa)


def in_interval(x, a, b):
    return np.logical_and(x >= np.minimum(a, b), x <= np.maximum(a, b))


def brent(f, a, b, xtol, maxiter=10):
    "Solve f(x) = 0, with a,b "
    fa = f(a)
    fb = f(b)
    assert np.all(fa*fb < 0), 'function not bracketed'
    a, b, fa, fb = sort(a, b, fa, fb)
    c, fc = a, fa
    mflag, d, s, fs = True, 0., None, None
    for iteration in range(maxiter):
        assert (fa == f(a)).all() and (fb == f(b)).all() and (fc == f(c)).all()
        assert s is None or (fs == f(s)).all()
        assert np.all(fa*fb <= 0), 'lost bracket'
        assert (np.abs(fa) >= np.abs(fb)).all()
        assert (np.abs(fa) >= np.abs(fc)).all()
        assert (np.abs(fc) >= np.abs(fb)).all()
        cq = converged(a, b, s, fa, fb, fs, xtol)
        if cq is not None:
            return cq
        s = quadratic(a, b, c, fa, fb, fc)
        s = np.where(np.isfinite(s), s, secant(a, b, fa, fb))
        u = np.where(mflag, np.abs(b-c), np.abs(c-d))
        cond1 = np.logical_not(in_interval(s, (3*a+b)/4, b))
        cond2_3 = abs(s-b) >= u/2
        cond4_5 = u < xtol
        mflag = cond1 | cond2_3 | cond4_5
        s = np.where(mflag, (a+b)*0.5, s)
        assert np.all(np.isfinite(s))
        fs = f(s)
        d, c = c, b
        fc = fb
        cond = fa*fs < 0
        b = np.where(cond, s, b)
        fb = np.where(cond, fs, fb)
        a = np.where(cond, a, s)
        fa = np.where(cond, fa, fs)
        a, b, fa, fb = sort(a, b, fa, fb)
    raise RuntimeError('iteration limit exceeded')
