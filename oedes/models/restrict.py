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

from oedes import model
from oedes.ad import *


class RestrictedModel(model):
    def __init__(self, model, eqs, x0, xt0):
        self.model = model
        self.idx = np.unique(np.hstack([model.findeq(n).idx for n in eqs]))
        self.bidx = np.zeros_like(model.X, dtype=np.bool)
        self.bidx[self.idx] = True
        self.nidx = np.arange(len(self.model.X))[np.logical_not(self.bidx)]
        self.dst = np.zeros_like(self.model.X, dtype=np.int)
        self.dst[self.idx] = np.arange(len(self.idx))
        self.X = np.zeros_like(self.idx, dtype=model.X.dtype)
        self.x0 = x0
        self.xt0 = xt0
        self.transientvar = model.transientvar[self.idx]
        self.eqs = eqs

    def _x_to_model(self, x, x0):
        return sparsesum(len(self.model.X), [
                         (self.idx, x), (self.nidx, x0[self.nidx])], unique=True, compress=False)

    def x_to_model(self, x):
        return self._x_to_model(x, self.x0)

    def evaluate(self, time, x, xt, params, **kwargs):
        model_X = self._x_to_model(x, self.x0)
        model_Xt = self._x_to_model(xt, self.xt0)
        for idx, f in self.model.evaluate(
                time, model_X, model_Xt, params, **kwargs):
            v = self.bidx[idx]
            yield self.dst[idx[v]], f[v]

    def update(self, x, params):
        model_X = self._x_to_model(x, self.x0)
        self.model.update(model_X, params)
        x[:] = model_X[self.idx]

    def scaling(self, params):
        xs, fs = self.model.scaling(params)
        return xs[self.idx], fs[self.idx]

    def converged(self, residuals, x, dx, params, report):
        model_residuals = self._x_to_model(residuals, 0. * self.model.X)
        model_x = self._x_to_model(x, 0. * self.model.X)
        model_dx = self._x_to_model(dx, 0. * self.model.X)
        report = {}
        r = self.model._converged(
            model_residuals,
            model_x,
            model_dx,
            params,
            report)
        return [report[k]['converged'] for k in self.eqs]
