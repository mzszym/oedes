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

__all__ = ['BuilderData', 'FVMBuilder', 'discretize']

import numpy as np
from oedes.utils import BuilderData
from collections import defaultdict
from .cell import FVMBoundaryEquation, FVMConservationEquation
from .poisson import FVMPoissonEquation
from .transport import FVMTransportEquation, FVMTransportChargedEquation
from .discrete import GeneralDiscreteEquation, DiscreteEquation, DummyConvergenceTest, DelegateConvergenceTest
from .evaluator import FVMEvaluator
from oedes.utils import ArgStack, EvaluationContext, as_computation
from .mesh import mesh


def _EquationBasedOrdering(equations):
    i = 0
    for eq in equations:
        n = eq.ndof()
        eq.idx = np.arange(i, i + n)
        i += n
    return i


def _CellBasedOrdering(equations):
    count = defaultdict(int)
    for eq in equations:
        if isinstance(eq, FVMConservationEquation):
            count[id(eq.mesh)] += 1
    j = dict()
    i = 0
    for eq in equations:
        n = eq.ndof()
        if not isinstance(eq, FVMConservationEquation):
            eq.idx = np.arange(i, i + n)
            i += n
        else:
            k = id(eq.mesh)
            if k not in j:
                j[k] = i
                i += eq.mesh.ncells * count[id(eq.mesh)]
            assert eq.ndof() == eq.mesh.ncells
            eq.idx = j[k] + np.arange(eq.mesh.ncells) * count[k]
            j[k] += 1
    return i


class FVMBuilder(BuilderData):
    def __init__(self, ordering='equation', mesh_mapper=None):
        super(FVMBuilder, self).__init__()
        if ordering == 'cell':
            indexer = _CellBasedOrdering
        elif ordering == 'equation':
            indexer = _EquationBasedOrdering
        else:
            raise ValueError('unknown equation order - %r' % ordering)
        self.indexer = indexer
        self.mesh_mapper = mesh_mapper

    def getMesh(self, key):
        if isinstance(key, mesh):
            return key
        return self.mesh_mapper(key)

    def setUp(self, context_type):
        self.ndof = self.indexer(self.equations)
        self.initEquations()
        self.evaluator = FVMEvaluator(self, context_type)
        self.finalize(self.evaluator)

    def newPoissonEquation(self, *args, **kwargs):
        return FVMPoissonEquation(*args, **kwargs)

    def newTransportEquation(self, *args, **kwargs):
        return FVMTransportEquation(*args, **kwargs)

    def newTransportChargedEquation(self, *args, **kwargs):
        return FVMTransportChargedEquation(*args, **kwargs)

    def newDirichletBC(self, *args, **kwargs):
        return FVMBoundaryEquation(*args, **kwargs)

    def newGeneralDiscreteEquation(self, *args, **kwargs):
        return GeneralDiscreteEquation(*args, **kwargs)

    def add(self, owner, eq):
        assert isinstance(eq, DiscreteEquation)
        assert owner is not None
        if isinstance(eq, FVMBoundaryEquation):
            assert eq.convergenceTest is not None
        super(FVMBuilder, self).add(owner, eq)

    def newDelegateConvergenceTest(self, obj):
        if obj is None:
            return DummyConvergenceTest()
        else:
            return DelegateConvergenceTest(obj)


def _meshdict(arg):
    if isinstance(arg, mesh):
        d = {None: arg}
    else:
        d = arg
    return lambda x: d[x]


def discretize(obj, mesh=None, **kwargs):
    obj = as_computation(obj)
    builder = FVMBuilder(mesh_mapper=_meshdict(mesh), **kwargs)
    obj.discretize(builder)
    builder.setUp(EvaluationContext)
    for _, eq in obj.all_equations(ArgStack()):
        eq.idx = builder.get(eq).idx  # TODO:
    return builder.evaluator
