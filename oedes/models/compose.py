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

__all__ = ['CompositeModel', 'With', 'Coupled']

from oedes import model
from oedes.fvm import discretize
from oedes.utils import With, Coupled
from .solver import SolverObject


class ModelAdapter(model):
    def __init__(self):
        super(ModelAdapter, self).__init__()
        self.discrete_model = None
        self.ordering = 'equation'

    def scaling(self, params):
        return self.discrete_model.scaling(params)

    def update(self, x, params):
        return self.discrete_model.update(x, params)

    def converged(self, residuals, x, dx, params, report):
        return self.discrete_model.converged(residuals, x, dx, params, report)

    @property
    def X(self):
        return self.discrete_model.X

    @property
    def transientvar(self):
        return self.discrete_model.transientvar

    def evaluate(self, time, x, xt, params, full_output=None, solver=None):
        if solver is None:
            solver = SolverObject()
        return self.discrete_model.evaluate(
            time, x, xt, params, full_output=full_output, solver=solver)

    def setUp(self, *args, **kwargs):
        assert self.discrete_model is None, 'already set-up'
        self.discrete_model = discretize(
            self, ordering=self.ordering, *args, **kwargs)


class CompositeModel(ModelAdapter, Coupled):
    def __init__(self, parts=None):
        super(CompositeModel, self).__init__()
        self.sub = []
        self.other = []
        if parts is not None:
            self.other.extend(parts)

    @property
    def parts(self):
        for sub in self.sub:
            yield With(sub, name=sub.name)
        for other in self.other:
            yield other
