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

from . import ad
import numpy as np
import scipy.sparse
import warnings


class model(object):
    X = None
    transientvar = None

    def evaluate(self, time, x, xt, params, full_output=None, solver=None):
        raise NotImplemented()

    def residuals(self, time, x, xt, params, full_output=None, solver=None):
        return ad.sparsesum(len(self.X), self.evaluate(
            time, x, xt, params, full_output=full_output, solver=solver))

    def output(self, time, x, xt, params, solver=None):
        output = {}
        for _ in self.evaluate(time, x, xt, params,
                               full_output=output, solver=solver):
            pass
        return output

    def converged(self, residuals, x, dx, params, report):
        raise NotImplemented()

    def update(self, x, params):
        pass

    def scaling(self, params):
        return 1., 1.
