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

from .dos import DOS
from oedes.ad import *
import scipy.constants
import oedes.functions.egdm as _egdm
import oedes.functions.gdos as _gdos
from oedes.functions.physics import ThermalVoltage
import numpy as np
import warnings


def egdm_params_simple(params, prefix, nsigma, a, mu0t):
    params[prefix + '.N0'] = _egdm.atoN0(a)
    params[
        prefix + '.sigma'] = _egdm.nsigma_inverse(nsigma, ThermalVoltage(params['T']))
    params[prefix + '.mu0'] = _egdm.mu0t_inverse(nsigma, mu0t)


class GaussianDOS(DOS):
    # it is allowed to turn off any enhancement factor
    use_g1 = True
    use_g2 = True
    use_g3 = True
    # to avoid numeric problems, c is clipped to more than c_eps
    c_eps = 1e-30
    c_max = 0.9999
    # custom limits are allowed
    g1_max_c = _egdm.g1_max_c
    g2_max_En = _egdm.g2_max_En
    g3_max_c = _egdm.g3_max_c
    # custom gdos impl is allowed, but must be compatible with
    # limits set here
    gdosImpl = _gdos.defaultImpl

    # To avoid numeric problems, nsigma is clipped to these values.
    # This can be useful in initial Newton iterations with
    # thermal coupling when temperature can be weird, and so nsigma.
    # However, final solution should not be affected by that.
    # Keep in mind that the model is for nsigma=3,4,5,6. nsigma<=1
    # will cause NaN in g1, and nsigma>>6 is problematic for Gauss-
    # Fermi integral.
    nsigma_min = 2.
    nsigma_max = 7.

    def __init__(self, impl=None):
        DOS.__init__(self)
        if impl is None:
            impl = _gdos.defaultImpl
        self.impl = impl

    def _load(self, ctx, eq):
        sigma = ctx.param(eq, 'sigma')
        Vt = ctx.varsOf(eq.thermal)['Vt']
        nsigma = _egdm.nsigma(sigma, Vt)
        nsigma = where(nsigma > self.nsigma_min, nsigma, self.nsigma_min)
        nsigma = where(nsigma < self.nsigma_max, nsigma, self.nsigma_max)
        if np.any(_egdm.nsigma(sigma, Vt) != nsigma):
            warnings.warn(
                'nsigma clipped to ensure numerical stability: probably not calculating what you want. This is OK in initial Newton iterations, but not in the final one')
        N0 = ctx.param(eq, 'N0')
        c = ctx.varsOf(eq)['c'] / N0
        assert self.c_eps >= self.impl.I_min
        assert self.c_max <= self.impl.I_max
        c = where(c > self.c_eps, c, self.c_eps)
        c = where(c < self.c_max, c, self.c_max)
        if 'dos_data' not in ctx.varsOf(eq):
            ctx.varsOf(eq)['dos_data'] = dict(b=self.impl.b(
                np.sqrt(2.) * nsigma, c))
        return sigma, nsigma, N0, Vt, c

    def Ef(self, ctx, eq):
        _, nsigma, N0, Vt, c = self._load(ctx, eq)
        return -ctx.varsOf(eq)['dos_data']['b'] * Vt

    def c(self, ctx, eq, Ef):
        _, nsigma, N0, Vt, c = self._load(ctx, eq)
        return N0 * self.impl.I(np.sqrt(2.) * nsigma, b=-Ef / Vt)

    def D(self, ctx, eq, c, Ef):
        raise NotImplementedError()

    def v_D(self, ctx, eq):
        sigma, nsigma, N0, Vt, c = self._load(ctx, eq)
        a = _egdm.N0toa(N0)
        mu_cell = _egdm.mu0t(nsigma, ctx.param(eq, 'mu0'))
        if self.use_g1:
            mu_cell = mu_cell * \
                _egdm.g1(nsigma, where(c < self.g1_max_c, c, self.g1_max_c))
        mu_face = eq.mesh.faceaverage(mu_cell)
        E = ctx.varsOf(eq.poisson)['E']
        if self.use_g2:
            En = (E * a) / sigma
            En2 = En**2
            En2_max = self.g2_max_En**2
            mu_face = mu_face * \
                _egdm.g2_En2(nsigma, where(En2 < En2_max, En2, En2_max))
        v = eq.v(mu_face, E)
        D_face = mu_face * Vt
        if self.use_g3:
            assert self.g3_max_c <= self.gdosImpl.I_max
            assert self.c_eps >= self.gdosImpl.I_min
            assert self.nsigma_max * np.sqrt(2.) <= self.gdosImpl.a_max
            inlimit_g3 = c < self.g3_max_c
            b_g3 = branch(inlimit_g3,
                          lambda ix: getitem(ctx.varsOf(eq)['dos_data']['b'],
                                             ix),
                          lambda ix: self.impl.b(np.sqrt(2.) * getitem(nsigma,
                                                                       ix),
                                                 self.g3_max_c))
            c_g3 = where(inlimit_g3, c, self.g3_max_c)
            D_face = D_face * \
                eq.mesh.faceaverage(
                    _egdm.g3(
                        nsigma,
                        c_g3,
                        impl=self.impl,
                        b=b_g3))
        return eq.v(mu_face, E), D_face
