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

from .simple import *
from .boundary import *
from oedes.ad import where, nvalue
import numpy as np
import logging

from oedes import logs


class DOS(object):

    def Ef(self, ctx, eq):
        "Return band-referenced Fermi level from concentration"
        raise NotImplementedError()

    def c(self, ctx, eq, Ef):
        "Return concentration from band-referenced Fermi level"
        raise NotImplementedError()

    def D(self, ctx, eq, c, Ef):
        "Return diffusion coefficient"
        raise NotImplementedError()

    def QuasiFermiLevel(self, ctx, eq):
        assert eq.z in [1, -1]
        Eband = -ctx.varsOf(eq.poisson)['potential'] - ctx.param(eq, 'level')
        return Eband - eq.z * self.Ef(ctx, eq)

    def concentration(self, ctx, eq, idx, imref):
        assert eq.z in [1, -1]
        Eband = - \
            ctx.varsOf(eq.poisson)['potential'][idx] - ctx.param(eq, 'level')
        return self.c(ctx, eq, -(imref - Eband) / eq.z)


class BoltzmannDOS(DOS):
    c_eps = 1e-30
    c_limit = True
    Ef_max = None
    logger = logs.models.getChild('BoltzmannDOS')

    def Ef(self, ctx, eq):
        c = ctx.varsOf(eq)['c']
        c_raw = c
        c = where(c > self.c_eps, c, self.c_eps)  # avoid NaN
        if self.logger.isEnabledFor(logging.INFO):
            if np.any(nvalue(c_raw) != nvalue(c)):
                self.logger.info(
                    'Ef(%r): clipping c<%r, min(c)=%r' %
                    (eq.prefix, self.c_eps, np.amin(
                        nvalue(c_raw))))
        N0 = ctx.param(eq, 'N0')
        return ctx.varsOf(eq.thermal)['Vt'] * np.log(c / N0)

    def c(self, ctx, eq, Ef):
        Ef_raw = Ef
        if self.Ef_max is not None:
            Ef = where(Ef < self.Ef_max, Ef, self.Ef_max)
            if self.logger.isEnabledFor(logging.INFO):
                if np.any(nvalue(Ef) != nvalue(Ef_raw)):
                    self.logger.info(
                        'c(%r): clipping Ef>%r, max(Ef)=%r' %
                        (eq.prefix, self.Ef_max, np.amax(
                            nvalue(Ef_raw))))
        N0 = ctx.param(eq, 'N0')
        v = Ef / ctx.varsOf(eq.thermal)['Vt']
        if self.c_limit:
            v_raw = v
            v = where(v <= 0., v, 0.)
            if self.logger.isEnabledFor(logging.INFO):
                if np.any(nvalue(v_raw) != nvalue(v)):
                    self.logger.info(
                        'c(%r): clipping Ef/kT>0, max(Ef/kT)=%r' %
                        (eq.prefix, np.amax(
                            nvalue(v_raw))))
        c = np.exp(v) * N0
        return c

    def D(self, ctx, eq, c, Ef):
        return ctx.varsOf(eq.thermal)['Vt']

    def v_D(self, ctx, eq):
        return species_v_D_charged_from_params(ctx, eq)
