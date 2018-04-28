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

from oedes import ad
import numpy as np
import scipy.special
import scipy.integrate
import scipy.interpolate
import scipy.version
from packaging.version import Version

if Version(scipy.version.version) < Version('0.19.0'):
    class make_interp_spline:
        def __init__(self, x, y, k=3):
            self.obj = scipy.interpolate.UnivariateSpline(x, y, s=0, k=k)

        def __call__(self, x, nu=0):
            if np.asarray(x).dtype == np.longdouble:
                x = np.asarray(x, dtype=np.double)
            return self.obj.__call__(x, nu=nu)
else:
    make_interp_spline = scipy.interpolate.make_interp_spline


class GaussFermiBase_(object):
    """
    Abstract implementation of Gauss-Fermi integral:
    I(a,b) = 1/sqrt(pi) \int_{-\infty}{+\infty} dx exp(-x^2)/(1.+exp(a*x+b))

    For simulations, three functions have to be provided in a consistent way:
    - forward calculation, provided by method I(a,b)
    - inverse calculation solving I(a,b)=I for b, provided by method b(a,I)
    - derivative calculation dI/db, provided by method dIdb(a,b)

    The last one is necessary because sparsegrad support is limited to one order
    of differentiation, therefore a function calculating first derivative has to
    be explicitly called to enable automatic calculation of second-order
    derivative.

    It is quite hard to give practical approximation of the integral which would
    work for all values. Therefore range of supported parameters values is
    also specified:
    - 0<=a<=a_max (for all calculations)
    - I_min<=I<=I_max (for calculation of b solving I(a,b)=I)

    Functions with underscore operate on internal representation of I, v=f(I).
    """

    maxiter = 20
    a_max = None
    I_min = None
    I_max = None
    atol = 0.
    rtol = 0.

    def _nevaluate(self, a, b, need_value=False, need_da=False,
                   need_db=False, need_d2aa=False, need_d2ab=False, need_d2bb=False):
        """
        Evaluate F (if need_value), and its derivatives (for example, dF/da if need_da and return 'da').
        """
        raise NotImplementedError()

    def _halley(self, function, b0, tol, maxiter):
        """
        Solve equation F(a,b)=i for b. function should return F, F', F'' or 0.
        """
        b = b0.copy()
        for itno in range(maxiter):
            f, df, d2f = function(b)
            if np.all(np.abs(f) < tol):
                return b
            delta = 2. * f * df / (2. * df * df - f * d2f)
            if not np.isfinite(delta).all():
                raise RuntimeError('numeric error')
            b -= delta
        raise RuntimeError('iteration limit exceeded')

    def _guess_b(self, a, v):
        raise NotImplementedError()

    def _nsolve_b(self, a, v, b0=None):
        tol = self.rtol * v + self.atol
        if b0 is None:
            b0 = self._guess_b(a, v)

        def function(b):
            e = self._nevaluate(
                a,
                b,
                need_value=True,
                need_db=True,
                need_d2bb=True)
            return e['value'] - v, e['db'], e['d2bb']
        return self._halley(function, b0, tol, self.maxiter)

    def _value(self, a, b):
        v = self._nevaluate(
            ad.nvalue(a), ad.nvalue(b), need_value=True, need_da=isinstance(
                a, ad.forward.value), need_db=isinstance(
                b, ad.forward.value))

        def _I(a, b):
            return v['value']

        def _I_deriv(args, value):
            yield lambda: v['da']
            yield lambda: v['db']
        return ad.custom_function(_I, _I_deriv)(a, b)

    def _dvdb(self, a, b):
        v = self._nevaluate(
            ad.nvalue(a),
            ad.nvalue(b),
            need_value=False,
            need_db=True,
            need_d2ab=isinstance(
                a,
                ad.forward.value),
            need_d2bb=isinstance(
                b,
                ad.forward.value))

        def _dIdb(a, b):
            return v['db']

        def _dIdb_deriv(args, value):
            yield lambda: v['d2ab']
            yield lambda: v['d2bb']
        return ad.custom_function(_dIdb, _dIdb_deriv)(a, b)

    def _b(self, a, v, b0=None):
        b = self._nsolve_b(ad.nvalue(a), ad.nvalue(v), b0=b0)

        def _b(a, v):
            return b

        def _b_deriv(args, value):
            a_, i_ = args
            b = value
            values = self._nevaluate(
                a_, b, need_da=isinstance(
                    a, ad.forward.value), need_db=True)
            dbdI = 1. / values['db']
            yield lambda: -dbdI * values['da']
            yield lambda: dbdI
        return ad.custom_function(_b, _b_deriv)(a, v)

    def __call__(self, a, b):
        return self.I(a, b)


class GaussFermiBase(GaussFermiBase_):
    rtol = 1e-14
    atol = 0.

    def _guess_b(self, a, i):
        # 1e-20 is added to avoid NaN, has no effect on calculation
        return np.where(i < 0.1, -np.log(i) + 0.25 * a **
                        2, np.log(1.0 / (i + 1e-20) - 1.))

    def I(self, a, b):
        return self._value(a, b)

    def dIdb(self, a, b):
        return self._dvdb(a, b)

    def b(self, a, i, b0=None):
        return self._b(a, i, b0=b0)


class GaussFermiLogBase(GaussFermiBase_):
    atol = 1e-13
    rtol = 0.

    def I(self, a, b):
        v = self._value(a, b)
        l_max = np.log(self.I_max)
        v = ad.where(v <= l_max, v, l_max)
        return ad.exp(v)

    def b(self, a, i, b0=None):
        return self._b(a, ad.log(i), b0=b0)

    def dIdb(self, a, b):
        v = self._nevaluate(
            ad.nvalue(a), ad.nvalue(b), need_value=True, need_da=isinstance(
                a, ad.forward.value), need_db=True, need_d2ab=isinstance(
                a, ad.forward.value), need_d2bb=isinstance(
                b, ad.forward.value))
        i = ad.exp(v['value'])

        def _dIdb(a, _):
            return i * v['db']

        def _dIdb_deriv(args, value):
            yield lambda: i * (v['d2ab'] + v['da'] * v['db'])
            yield lambda: i * (v['d2bb'] + v['db'] * v['db'])
        return ad.custom_function(_dIdb, _dIdb_deriv)(a, b)


class GaussFermiIntegralH(GaussFermiBase):
    """GaussFermiIntegral implemented using Gauss-Hermite quadrature"""

    def __init__(self, n, a_max):
        super(GaussFermiIntegralH, self).__init__()
        self.maxiter = 20
        self.a_max = a_max
        self.I_min = 1e-30
        self.I_max = 0.9999
        _x, _w = scipy.special.h_roots(n)
        self.X_W = list(zip(_x, _w / np.sqrt(np.pi)))

    def _nevaluate(self, a, b, need_value=False, need_da=False,
                   need_db=False, need_d2aa=False, need_d2ab=False, need_d2bb=False):
        I, da, db, d2aa, d2ab, d2bb = None, None, None, None, None, None
        if need_value:
            I = 0.
        if need_da:
            da = 0.
        if need_db:
            db = 0.
        if need_d2aa:
            d2aa = 0.
        if need_d2ab:
            d2ab = 0.
        if need_d2bb:
            d2bb = 0.
        need_first = da is not None or db is not None
        need_second = d2aa is not None or d2ab is not None or d2bb is not None
        need_deriv = need_first or need_second
        for x, w in self.X_W:
            axpb = a * x + b
            u = np.exp(axpb)
            f = 1. / (1. + u)
            if I is not None:
                I += w * f
            if need_deriv:
                u_f = np.where(axpb < 40., u * f, 1.)
                df = -f * u_f
                w_df = w * df
                if need_first:
                    if da is not None:
                        da += w_df * x
                    if db is not None:
                        db += w_df
                if need_second:
                    w_ddf = w_df * (1. - 2. * u_f)
                    if d2aa is not None:
                        d2aa += w_ddf * x * x
                    if d2ab is not None:
                        d2ab += w_ddf * x
                    if d2bb is not None:
                        d2bb += w_ddf
        return dict(value=I, da=da, db=db, d2aa=d2aa, d2ab=d2ab, d2bb=d2bb)


class GaussFermiIntegralReference(GaussFermiBase):
    """
    Reference implementation for checking accuracy, which uses adaptive
    quadrature.
    Only calculation of I is supported.
    """

    def _nevaluate(self, a, b, **kwargs):
        need_value = kwargs.pop('need_value', False)
        if any(kwargs.values()):
            raise NotImplementedError()

        def _f(x):
            return np.exp(-x * x) / (1 + np.exp(a * x + b))
        v = scipy.integrate.quad(_f, -28, 28,
                                 epsrel=1e-12, epsabs=1e-40)
        return dict(value=v[0] / np.sqrt(np.pi))


class UnivariateInterpolatedGaussFermi(GaussFermiLogBase):
    def __init__(self, impl, a, n=1000, k=5):
        super(UnivariateInterpolatedGaussFermi, self).__init__()
        self.a = a
        self.a_max = impl.a_max
        self.I_min = impl.I_min
        self.I_max = impl.I_max
        self.b_min, self.b_max = map(
            lambda b: impl.b(
                a, b), (impl.I_min, impl.I_max))
        self.btab = np.linspace(self.b_min, self.b_max, n)
        _i = impl.I(ad.forward.seed(a), self.btab)
        self.logItab = np.log(ad.nvalue(_i))
        self.dlogIdatab = 1. / ad.nvalue(_i) * _i.gradient.tocsr().todense().A1
        assert np.all(np.diff(self.logItab) > 0)
        self.spline = make_interp_spline(
            self.btab[::-1], self.logItab[::-1], k=k)
        self.daspline = make_interp_spline(
            self.btab[::-1], self.dlogIdatab[::-1], k=k)

    def _guess_b(self, a, v):
        return self.btab[np.clip(np.searchsorted(
            self.logItab, v), 0, len(self.btab) - 1)]

    def _nevaluate(self, a, b, **tasks):
        if np.any(a != self.a):
            raise ValueError('a is different than precalculated')
        if np.asarray(a).shape == (0,):
            assert np.asarray(b).shape in [(), (0,)]
            b = np.zeros(0, dtype=np.common_type(a, b))
        result = {}
        if tasks.pop('need_value', False):
            result['value'] = self.spline(b)
        if tasks.pop('need_db', False):
            result['db'] = self.spline(b, 1)
        if tasks.pop('need_d2bb', False):
            result['d2bb'] = self.spline(b, 2)
        if tasks.pop('need_da', False):
            result['da'] = self.daspline(b)
        if tasks.pop('need_d2ab', False):
            result['d2ab'] = self.daspline(b, 1)
        if any(tasks.values()):
            raise NotImplementedError()
        return result


class UnivariateInterpolatedGaussFermiFactory(object):
    def __init__(self, impl, **kwargs):
        self.I_min = impl.I_min
        self.I_max = impl.I_max
        self.a_max = impl.a_max
        self.impl = impl
        self.kwargs = kwargs
        self.d = {}

    def _get(self, a):
        a = ad.nvalue(a)
        if a.shape:
            if len(a) > 0:
                if np.count_nonzero(a != a[0]) > 0:
                    raise ValueError('a must have all values equal')
                a = a[0]
            else:
                # a is empty: return anything
                if self.d:
                    return next(iter(self.d.values()))
                else:
                    return self._get(0)
        a = float(a)
        u = self.d.get(a, None)
        if u is None:
            u = UnivariateInterpolatedGaussFermi(
                self.impl, np.asarray(a), **self.kwargs)
            self.d[a] = u
        return u

    def I(self, a, b):
        return self._get(a).I(a, b)

    def b(self, a, I):
        return self._get(a).b(a, I)

    def dIdb(self, a, b):
        return self._get(a).dIdb(a, b)


def diffusion_enhancement(nsigma_as_a, c, impl, b=None, full_output=False):
    if b is None:
        b = impl.b(nsigma_as_a, c)
    factor = -c / impl.dIdb(nsigma_as_a, b)
    if full_output:
        return dict(factor=factor, b=b)
    else:
        return factor


defaultImpl = GaussFermiIntegralH(96, 7. * np.sqrt(2.))
