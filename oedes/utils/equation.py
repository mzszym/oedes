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

from .computation import Computation
from .funcall import Funcall

_all__ = ['Equation', 'Calculation', 'EquationWithMesh', 'SubEquation']


class Equation(Computation):
    def __init__(self, name=None):
        super(Equation, self).__init__()

        self.name = name
        self.convergenceTest = None

    def all_equations(self, args):
        yield (args.push(name=self.name), self)

    def newDiscreteEq(self, builder):
        return builder.newGeneralDiscreteEquation(0)

    def initDiscreteEq(self, builder, obj):
        "Late initialization of discrete equation object (after all objects were created)"
        pass

    def build(self, builder, args):
        eq = self.newDiscreteEq(builder)
        eq.prefix = args.prefix
        self.buildDiscreteEq(builder, eq)
        builder.addInitializer(self.initDiscreteEq, eq)
        return eq

    def load(self, ctx, eq):
        vars = ctx.newVars(eq)
        vars['x'] = ctx.gvars['x'][eq.idx]
        vars['xt'] = ctx.gvars['xt'][eq.idx]

    def evaluate(self, ctx, eq):
        pass

    def buildDiscreteEq(self, builder, obj):
        obj.convergenceTest = builder.newDelegateConvergenceTest(self)
        obj.load = Funcall(self.load, obj)
        obj.evaluate = Funcall(self.evaluate, obj, depends=[obj.load])
        obj.alldone = Funcall(None, obj, depends=[obj.evaluate])
        builder.addEvaluation(obj.load)
        builder.addEvaluation(obj.evaluate)
        builder.addEvaluation(obj.alldone)
        builder.add(self, obj)


class SubEquation(object):
    def subInit(self, parent, builder, parent_eq):
        pass


class EquationWithMesh(Equation):
    def __init__(self, mesh, name=None):
        super(EquationWithMesh, self).__init__(name)
        self.mesh = mesh


class Calculation(Equation):
    def buildDiscreteEq(self, builder, obj):
        super(Calculation, self).buildDiscreteEq(builder, obj)
        obj.convergenceTest = builder.newDelegateConvergenceTest(None)
