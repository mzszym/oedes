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

__all__ = ['Builder', 'FVMBuilder']

import numpy as np
from collections import defaultdict
from oedes.utils import FuncallGraph
from .cell import FVMBoundaryEquation, FVMConservationEquation
from .poisson import FVMPoissonEquation
from .transport import FVMTransportEquation, FVMTransportChargedEquation
from .discrete import GeneralDiscreteEquation, DiscreteEquation
from .evaluator import FVMEvaluator


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


class Builder(object):
    pass


class FVMBuilder(Builder):
    def __init__(self, ordering='equation'):
        self.equations = []
        self.key_to_eq = dict()
        self.init_list = []
        self.fungraph = FuncallGraph()
        if ordering == 'cell':
            self.indexer = _CellBasedOrdering
        elif ordering == 'equation':
            self.indexer = _EquationBasedOrdering
        else:
            raise ValueError('unknown equation order - %r' % ordering)

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

    def getMesh(self, obj):
        return obj

    def add(self, owner, eq):
        assert isinstance(eq, DiscreteEquation)
        assert owner is not None
        if isinstance(eq, FVMBoundaryEquation):
            assert eq.convergenceTest is not None
        assert id(owner) not in self.key_to_eq
        self.equations.append(eq)
        self.key_to_eq[id(owner)] = eq

    def addInitializer(self, func, *args):
        self.init_list.append((func, args))

    def addEvaluation(self, func):
        self.fungraph.add(func)

    def setUp(self, context_type):
        self.ndof = self.indexer(self.equations)
        for func, args in self.init_list:
            func(self, *args)
        for eq in self.equations:
            eq.init(self)
        self.evaluator = FVMEvaluator(self, context_type)

    def get(self, owner):
        return self.key_to_eq[id(owner)]
