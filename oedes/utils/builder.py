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

from oedes.utils import FuncallGraph

__all__ = ['BuilderData']


class BuilderData(object):
    """
    Data used during construction of discrete model

    The discretization process proceeds as follows:

    1. For each calculation, discrete object is build. At this stage,
    calculations add added to graph using `addEvaluation`, and discrete
    objects add using `add`. Initialization that must be done on
    complete system is registered by `addInitializer`.

    2. After all objects are built, initializers registered using `addInitializer`
    are called. Also, `init` methods in discrete objects are called (`initEquations` functions)

    3. Evaluator is constructed using attributes.

    4. After evaluator is constructed, final initializers registered using `registerFinalizer`
    are called with (builder, model, *args)

    Attributes
    ----------
    equations : list
        All discrete objects
    fungraph : FuncallGraph
        Evaluation graph
    """

    def __init__(self):
        self.equations = []
        self.key_to_eq = dict()
        self.init_list = []
        self.finalizers = set()
        self.fungraph = FuncallGraph()

    def getMesh(self, key):
        "Return concrete mesh object for mesh handle"
        raise NotImplementedError()

    def get(self, owner):
        "Return discrete object for calculation object"
        return self.key_to_eq[id(owner)]

    def addInitializer(self, func, *args):
        """
        Add initializer

        func(*args) will be called after discrete objects are constructed for all calculations
        """
        self.init_list.append((func, args))

    def registerFinalizer(self, func, *args):
        """
        Register final initializer

        func(evaluator, *args) will be called after evaluator is constructed
        """
        self.finalizers.add((func, args))

    def addEvaluation(self, func):
        "Add function evaluation to discrete model"
        self.fungraph.add(func)

    def add(self, owner, eq):
        """
        Add discrete object `eq` corresponding to `owner`

        `eq` can is retreived by calling ``self.get(owner)``. Only one discrete equation can
        exist for `owner`
        """
        assert id(owner) not in self.key_to_eq
        self.equations.append(eq)
        self.key_to_eq[id(owner)] = eq

    def initEquations(self):
        "Run late initialization"
        for func, args in self.init_list:
            func(self, *args)
        for eq in self.equations:
            eq.init(self)

    def finalize(self, evaluator):
        for func, args in self.finalizers:
            func(self, evaluator, *args)
