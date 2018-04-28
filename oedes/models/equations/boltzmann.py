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

__all__ = ['BoltzmannDOS']

from oedes.ad import where, nvalue
import logging
import numpy as np
from oedes import logs
from .dos import DOS


class BoltzmannDOS(DOS):
    c_eps = 1e-30
    c_limit = True
    Ef_max = None
    logger = logs.models.getChild('BoltzmannDOS')

    def N0(self, ctx, eq):
        return ctx.param(eq, 'N0')

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
        N0 = self.N0(ctx, eq)
        return ctx.varsOf(eq.thermal)['Vt'] * np.log(c / N0)

    def Ef_correction(self, ctx, eq):
        return 0.

    def c(self, ctx, eq, Ef, numeric_parameters=False):
        Ef_raw = Ef
        if self.Ef_max is not None:
            Ef = where(Ef < self.Ef_max, Ef, self.Ef_max)
            if self.logger.isEnabledFor(logging.INFO):
                if np.any(nvalue(Ef) != nvalue(Ef_raw)):
                    self.logger.info(
                        'c(%r): clipping Ef>%r, max(Ef)=%r' %
                        (eq.prefix, self.Ef_max, np.amax(
                            nvalue(Ef_raw))))
        N0 = self.N0(ctx, eq)
        if numeric_parameters:
            N0 = nvalue(N0)
        Vt = ctx.varsOf(eq.thermal)['Vt']
        if numeric_parameters:
            Vt = nvalue(Vt)
        v = Ef / Vt
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

    def D_mu(self, parent_eq, ctx, eq):
        return ctx.varsOf(eq.thermal)['Vt']
