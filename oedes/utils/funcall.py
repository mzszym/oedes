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

__all__ = ['Funcall', 'FuncallGraph']

from collections import defaultdict
import weakref


class Funcall(object):
    def __init__(self, func, obj, depends=()):
        self.func = func
        self._obj = weakref.ref(obj)
        self._depends = list(depends)

    def depends(self, f):
        assert isinstance(f, Funcall)
        #assert f not in self._depends
        self._depends.append(f)

    def __hash__(self):
        return hash(id(self))

    def __eq__(self, other):
        return self is other

    def __call__(self, ctx):
        return self.func(ctx, self._obj())


class FuncallGraph(object):
    def __init__(self):
        self.evaluations = []

    def add(self, func):
        self.evaluations.append(func)

    def plan(self):
        evaluations = self.evaluations
        rdeps = defaultdict(list)
        queue = []
        for e in evaluations:
            assert isinstance(e, Funcall)
            for d in e._depends:
                assert isinstance(d, Funcall)
                rdeps[d].append(e)
            if not e._depends:
                queue.append(e)
        count = dict((f, len(f._depends)) for f in evaluations)
        while queue:
            t = queue.pop()
            yield t
            for r in rdeps[t]:
                count[r] -= 1
                if count[r] == 0:
                    queue.append(r)
        assert not any(count.values(
        )), 'unfinished tasks left: not added but registered as dependencies or dependence cycles'
