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

__all__ = ['CompositeModel', 'With', 'Coupled']

from oedes import model
from oedes.fvm import FVMBuilder
from .utils import EvaluationContext
from oedes.utils import With, ArgStack, Coupled
from .solver import SolverObject, RamoShockleyCalculation
from .import boundary
from oedes.solver import solve
import numpy as np


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

    def setUpRamoShockleyTestfunctions(
            self, dtype=np.double, solve_kwargs=None):
        if solve_kwargs is None:
            solve_kwargs = dict()
        rs_boundaries = set()
        for _, eq in self.all_equations(ArgStack()):
            if isinstance(eq, boundary.AppliedVoltage):
                rs_boundaries.add(eq.name)
        params = dict()
        for b in rs_boundaries:
            s = RamoShockleyCalculation(b, store=False)
            x = solve(
                self,
                np.asarray(
                    self.X,
                    dtype=dtype),
                params,
                niter=1,
                solver=s,
                **solve_kwargs)
            s = RamoShockleyCalculation(b, store=True)
            self.output(
                0,
                x,
                np.zeros_like(x),
                params,
                solver=s,
                **solve_kwargs)

    def setUp(self):
        assert self.discrete_model is None, 'already set-up'
        builder = FVMBuilder(ordering=self.ordering)
        self.discretize(builder)
        builder.setUp(EvaluationContext)
        for _, eq in self.all_equations(ArgStack()):
            eq.idx = builder.get(eq).idx  # TODO:
        self.discrete_model = builder.evaluator
        self.setUpRamoShockleyTestfunctions()


class CompositeModel(ModelAdapter, Coupled):
    def __init__(self):
        super(CompositeModel, self).__init__()
        self.sub = []
        self.other = []

    @property
    def parts(self):
        for sub in self.sub:
            yield With(sub, prefix=sub.name)
        for other in self.other:
            yield other
