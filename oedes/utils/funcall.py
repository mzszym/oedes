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
        return self.func(ctx, self.obj())

    def obj(self):
        obj = self._obj()
        assert obj is not None
        return obj


class FuncallGraph(object):
    def __init__(self):
        self.evaluations = []

    def add(self, func):
        self.evaluations.append(func)

    def plan(self, kill_func=None):
        evaluations = list(self.evaluations)
        rdeps = defaultdict(list)
        queue = list()

        # Build reverse dependence graph `rdeps`, and set of all calculations
        # `known`
        known = set(self.evaluations)
        while evaluations:
            e = evaluations.pop()
            assert isinstance(e, Funcall)
            for d in e._depends:
                assert isinstance(d, Funcall)
                rdeps[d].append(e)
                # if dependence was not added explicitely by calling `add`
                # method, add it here
                if d not in known:
                    known.add(d)
                    evaluations.append(d)
            if not e._depends:
                queue.append(e)

        waitcount = dict((f, len(f._depends)) for f in known)
        usecount_byobj = defaultdict(int)
        for f in known:
            for d in f._depends:
                usecount_byobj[id(d.obj())] += 1
        while queue:
            t = queue.pop()
            if t.func is not None:  # do not yield checkpoints where fun is None
                yield t
            if not usecount_byobj[id(t.obj())]:
                yield kill_func(t.obj())
            for d in t._depends:
                obj = d._obj()
                assert usecount_byobj[id(obj)] > 0
                usecount_byobj[id(obj)] -= 1
                if not usecount_byobj[id(obj)]:
                    if kill_func is not None:
                        yield kill_func(obj)
            for r in rdeps[t]:
                assert waitcount[r] >= 1
                waitcount[r] -= 1
                if waitcount[r] == 0:
                    queue.append(r)
        assert not any(waitcount.values(
        )), 'unfinished tasks left: not added but registered as dependencies or dependence cycles'
