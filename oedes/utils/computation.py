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

import itertools

__all__ = ['ArgStack', 'Computation', 'With', 'Coupled', 'as_computation']


class ArgStack(object):
    stack_keys = [('prefix', 'name')]

    def __init__(self, **kwargs):
        for field, _ in self.stack_keys:
            if field not in kwargs:
                kwargs[field] = []
        self.kwargs = kwargs
        self._prefix = '.'.join(p for p in self.kwargs['prefix'])

    def push(self, **kwargs):
        for field, _ in self.stack_keys:
            assert field not in kwargs
        args = dict(self.kwargs)
        args.update(kwargs)
        for field, key in self.stack_keys:
            if key in kwargs and kwargs[key] is not None:
                args[field] = self.kwargs[field] + [kwargs[key]]
        return ArgStack(**args)

    @property
    def prefix(self):
        return self._prefix

    def __getattr__(self, name):
        return self.kwargs[name]


class Computation(object):
    def all_equations(self, args):
        yield (args, self)

    def build(self, builder):
        raise NotImplementedError()

    def findeq(self, name):
        return dict((a.prefix, eq)
                    for a, eq in self.all_equations(ArgStack()))[name]

    def findeqs(self, test, return_names=False):
        for a, eq in self.all_equations(ArgStack()):
            if test(eq):
                if return_names:
                    yield a.prefix, eq
                else:
                    yield eq


def as_computation(obj):
    "Returns object as calculation"
    if isinstance(obj, Computation):
        return obj
    return Coupled(obj)


class With(Computation):
    "Assings arguments (name etc.) to calculations"

    def __init__(self, eq, **kwargs):
        self.eq = as_computation(eq)
        self.kwargs = kwargs

    def all_equations(self, args):
        defaults = args.push(**self.kwargs)
        return self.eq.all_equations(defaults)


class Coupled(Computation):
    "Coupled calculations"

    def __init__(self, parts=None):
        if parts is not None:
            self.parts = tuple(map(as_computation, parts))

    def all_equations(self, args):
        return itertools.chain(*(p.all_equations(args) for p in self.parts))

    def discretize(self, builder):
        for args, eq in self.all_equations(ArgStack()):
            eq.build(builder, args)
