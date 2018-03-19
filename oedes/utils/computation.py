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

import itertools

__all__ = ['ArgStack', 'Computation', 'With', 'Coupled', 'as_computation']


class ArgStack(object):
    stack_keys = ['prefix']

    def __init__(self, **kwargs):
        for k in self.stack_keys:
            if k not in kwargs:
                kwargs[k] = None
        self.kwargs = kwargs

    def push(self, **kwargs):
        for k in self.stack_keys:
            if k in kwargs:
                v = kwargs[k]
                if v is None:
                    kwargs[k] = self.kwargs[k]
                    continue
                if not v:
                    raise ValueError('empty prefix not allowed')
                # if '.' in v:
                #    raise ValueError('dot . not allowed in prefix name')
                if self.kwargs[k]:
                    kwargs[k] = '.'.join([self.kwargs[k], v])
                else:
                    kwargs[k] = v
        return ArgStack(**kwargs)

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
    "Assings arguments (prefix, domain) to calculations"

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
            obj = eq.build(builder)
            obj.new_prefix = args.prefix
