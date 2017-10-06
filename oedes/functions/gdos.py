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

from oedes import ad
import numpy as np
import scipy.special
import scipy.integrate


class AbstractGaussFermiIntegralImpl(object):
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
    """

    maxiter = 20
    a_max = None
    I_min = None
    I_max = None

    def nevaluate(self, a, b, need):
        """
        Perform numerical evaluation of needed quantities need with known
        parameters a,b. In need, 'I' denotes I(a,b), 'da' denotes dI/da,
        'db' denotes dI/db, 'd2ab' denotes d2I/(dadb) and 'd2bb' denotes
        d2I/db2.
        """
        raise NotImplementedError()

    def nsolve_b(self, a, i, b0=None, halley=True, tol=1e-14):
        """
        Solve equation I(a,b)=i for b with known a.
        """
        if b0 is None:
            # 1e-20 is added to avoid NaN, has no effect on calculation
            b = np.where(i < 0.1, -np.log(i) + 0.25 * a **
                         2, np.log(1.0 / (i + 1e-20) - 1.))
        else:
            b = b0
        need = ['I', 'db']
        if halley:
            need.append('d2bb')
        for itno in range(self.maxiter):
            values = self.nevaluate(a, b, need=need)
            I = values['I']
            df = values['db']
            if halley:
                d2f = values['d2bb']
            else:
                d2f = 0.
            f = I - i
            if np.amax(np.abs(f) / i) < tol:
                return b
            delta = 2. * f * df / (2. * df * df - f * d2f)
            if not np.isfinite(delta).all():
                raise RuntimeError('numeric error')
            b -= delta
        raise RuntimeError('iteration limit exceeded')

    def I(self, a, b):
        def _I(a, b):
            return self.nevaluate(a, b, need=['I'])['I']

        def _I_deriv(args, value):
            a, b = args
            values = self.nevaluate(a, b, need=['da', 'db'])
            yield lambda: values['da']
            yield lambda: values['db']
        return ad.custom_function(_I, _I_deriv)(a, b)

    def dIdb(self, a, b):
        def _dIdb(a, b):
            return self.nevaluate(a, b, need=['db'])['db']

        def _dIdb_deriv(args, value):
            a, b = args
            values = self.nevaluate(a, b, need=['d2ab', 'd2bb'])
            yield lambda: values['d2ab']
            yield lambda: values['d2bb']
        return ad.custom_function(_dIdb, _dIdb_deriv)(a, b)

    def b(self, a, i, b0=None):
        def _b(a, i):
            return self.nsolve_b(a, i, b0=b0, halley=True)

        def _b_deriv(args, value):
            a, i = args
            b = value
            values = self.nevaluate(a, b, need=['da', 'db'])
            dIda, dIdb = values['da'], values['db']
            dbdI = 1. / dIdb
            yield lambda: -dbdI * dIda
            yield lambda: dbdI
        return ad.custom_function(_b, _b_deriv)(a, i)

    def __call__(self, a, b):
        return self.I(a, b)


class GaussFermiIntegralH(AbstractGaussFermiIntegralImpl):
    """GaussFermiIntegral implemented using Gauss-Hermite quadrature"""

    def __init__(self, n, a_max):
        AbstractGaussFermiIntegralImpl.__init__(self)
        self.maxiter = 20
        self.a_max = a_max
        self.I_min = 1e-30
        self.I_max = 0.9999
        _x, _w = scipy.special.h_roots(n)
        self.X_W = list(zip(_x, _w / np.sqrt(np.pi)))

    def nevaluate(self, a, b, need):
        I, da, db, d2ab, d2bb = None, None, None, None, None
        if 'I' in need:
            I = 0.
        if 'da' in need:
            da = 0.
        if 'db' in need:
            db = 0.
        if 'd2ab' in need:
            d2ab = 0.
        if 'd2bb' in need:
            d2bb = 0.
        need_first = da is not None or db is not None
        need_second = d2ab is not None or d2bb is not None
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
                    if d2ab is not None:
                        d2ab += w_ddf * x
                    if d2bb is not None:
                        d2bb += w_ddf
                    # nb: d2aa=w_ddf*x*x
        loc = locals()
        return dict((k, loc[k]) for k in need)


class GaussFermiIntegralReference(AbstractGaussFermiIntegralImpl):
    """
    Reference implementation for checking accuracy, which uses adaptive
    quadrature.
    Only calculation of I is supported.
    """

    def nevaluate(self, a, b, need):
        if not need:
            return dict()
        if list([v for v in need if v != 'I']):
            raise NotImplementedError()

        def _f(x):
            return np.exp(-x * x) / (1 + np.exp(a * x + b))
        v = scipy.integrate.quad(_f, -28, 28,
                                 epsrel=1e-12, epsabs=1e-40)
        return dict(I=v[0] / np.sqrt(np.pi))


def diffusion_enhancement(nsigma_as_a, c, impl, full_output=False):
    b = impl.b(nsigma_as_a, c)
    factor = -c / impl.dIdb(nsigma_as_a, b)
    if full_output:
        return dict(factor=factor, b=b)
    else:
        return factor


defaultImpl = GaussFermiIntegralH(96, 7. * np.sqrt(2.))
