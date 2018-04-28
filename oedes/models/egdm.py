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

from oedes.models.equations import DOS, MobilityModel
from oedes.utils import SubEquation
from oedes.ad import where, branch, getitem, nvalue, log
import oedes.functions.egdm as _egdm
import oedes.functions.gdos as _gdos
from oedes.functions.physics import ThermalVoltage
import numpy as np
from oedes import logs


def egdm_params_simple(params, prefix, nsigma, a, mu0t):
    params[prefix + '.N0'] = _egdm.atoN0(a)
    params[
        prefix + '.sigma'] = _egdm.nsigma_inverse(nsigma, ThermalVoltage(params['T']))
    params[prefix + '.mu0'] = _egdm.mu0t_inverse(nsigma, mu0t)


class DisorderFromParams(SubEquation):
    def disorder_parameters(self, parent, ctx, eq):
        info = dict(sigma=ctx.param(eq, 'sigma'),
                    mu0=ctx.param(eq, 'mu0'),
                    Vt=ctx.varsOf(eq.thermal)['Vt'],
                    N0=ctx.param(eq, 'N0'))
        info['nsigma'] = _egdm.nsigma(sigma=info['sigma'], Vt=info['Vt'])
        return info


class GaussianDOS(DOS):
    # it is allowed to turn off any enhancement factor
    use_g3 = True
    # to avoid numeric problems, c is clipped to more than c_eps
    c_eps = 1e-30
    c_max = 0.9999
    # custom limits are allowed
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

    logger = logs.models.getChild('GaussianDOS')

    def __init__(self, impl=None, disorder=None):
        super(GaussianDOS, self).__init__()
        if impl is None:
            impl = _gdos.defaultImpl
        self.impl = impl
        if disorder is None:
            disorder = DisorderFromParams()
        self.disorder = disorder

    def _load(self, ctx, eq):
        variables = ctx.varsOf(eq)
        if 'gdos_data' in variables:
            return variables['gdos_data']
        info = self.disorder.disorder_parameters(self, ctx, eq)
        nsigma = info['nsigma']
        nsigma = where(nsigma > self.nsigma_min, nsigma, self.nsigma_min)
        nsigma = where(nsigma < self.nsigma_max, nsigma, self.nsigma_max)
        if np.any(nsigma != info['nsigma']):
            self.logger.info(
                'nsigma clipped to ensure numerical stability: probably not calculating what you want. This is OK in initial Newton iterations, but not in the final one')
        N0 = info['N0']
        c = ctx.varsOf(eq)['c'] / N0
        assert self.c_eps >= self.impl.I_min
        assert self.c_max <= self.impl.I_max
        c = where(c > self.c_eps, c, self.c_eps)
        c = where(c < self.c_max, c, self.c_max)
        b = self.impl.b(np.sqrt(2.) * nsigma, c)
        info.update(b=b, c=c, nsigma=nsigma)
        variables['gdos_data'] = info
        return info

    def Ef(self, ctx, eq):
        info = self._load(ctx, eq)
        return -info['b'] * info['Vt']

    def Ef_correction(self, ctx, eq):
        info = self._load(ctx, eq)
        return (-info['b'] - log(info['c'])) * info['Vt']

    def c(self, ctx, eq,  Ef, numeric_parameters=False):
        info = self._load(ctx, eq)
        N0 = info['N0']
        nsigma = info['nsigma']
        Vt = info['Vt']
        if numeric_parameters:
            N0 = nvalue(N0)
            nsigma = nvalue(nsigma)
            Vt = nvalue(Vt)
        return N0 * self.impl.I(np.sqrt(2.) * nsigma, b=-Ef / info['Vt'])

    def D_mu(self, parent, ctx, eq):
        info = self._load(ctx, eq)
        assert self.g3_max_c <= self.gdosImpl.I_max
        assert self.c_eps >= self.gdosImpl.I_min
        assert self.nsigma_max * np.sqrt(2.) <= self.gdosImpl.a_max
        c = info['c']
        nsigma = info['nsigma']
        inlimit_g3 = c < self.g3_max_c

        def branch_if_inlimit(ix):
            return getitem(info['b'], ix)

        def branch_cut_limit(ix):
            return self.impl.b(
                np.sqrt(2.) * getitem(nsigma, ix), self.g3_max_c)
        b_g3 = branch(inlimit_g3, branch_if_inlimit, branch_cut_limit)
        c_g3 = where(inlimit_g3, c, self.g3_max_c)
        g3 = _egdm.g3(nsigma, c_g3, impl=self.impl, b=b_g3)
        return eq.mesh.faceaverage(g3*info['Vt'])


class EGDMMobility(MobilityModel):
    g1_max_c = _egdm.g1_max_c
    g2_max_En = _egdm.g2_max_En

    def __init__(self, use_g1=True, use_g2=True):
        super(EGDMMobility, self).__init__()
        self.use_g1 = use_g1
        self.use_g2 = use_g2

    def _load(self, ctx, eq):
        return ctx.varsOf(eq)['gdos_data']

    def mobility(self, parent, ctx, eq):
        info = self._load(ctx, eq)
        a = _egdm.N0toa(info['N0'])
        nsigma = info['nsigma']
        sigma = info['sigma']
        mu_cell = _egdm.mu0t(nsigma, info['mu0'])
        c = info['c']
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
        ctx.varsOf(eq).update(
            mu_face=mu_face,
            mu_cell=eq.mesh.cellaveragev(mu_face))
