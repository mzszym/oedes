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

__all__ = ['WeakList']

import weakref


def _unwrap(item):
    assert isinstance(item, weakref.ReferenceType)
    return item()


def _wrap(obj):
    assert not isinstance(obj, weakref.ReferenceType)
    return weakref.ref(obj)


class WeakList(list):
    """List of weak references emulating normal list"""

    def __init__(self, items=None):
        if items is None:
            items = []
        super(WeakList, self).__init__(map(_wrap, items))

    def __contains__(self, obj):
        return super(WeakList, self).__contains__(self, _wrap(obj))

    def __getitem__(self, index):
        result = super(WeakList, self).__getitem__(index)
        if isinstance(result, list):
            return list(map(_unwrap, result))
        return _unwrap(result)

    def __setitem__(self, index, item):
        if isinstance(index, slice):
            super(WeakList, self).__setitem__(index, map(_wrap, item))
        else:
            return _wrap(item)

    def __iter__(self):
        itr = super(WeakList, self).__iter__()
        return iter(_unwrap(x) for x in itr)

    def __reversed__(self):
        result = type(self)(self)
        result.reverse()
        return result

    def __iadd__(self, other):
        self.extend(WeakList(other))
        return self

    def __eq__(self, other):
        return list(self) == other

    def __neq__(self, other):
        return list(self) != other

    def __lt__(self, other):
        return list(self) < other

    def __le__(self, other):
        return list(self) <= other

    def __gt__(self, other):
        return list(self) > other

    def __ge__(self, other):
        return list(self) >= other

    def append(self, item):
        super(WeakList, self).append(_wrap(item))

    def remove(self, item):
        super(WeakList, self).remove(_wrap(item))

    def index(self, item):
        return super(WeakList, self).index(_wrap(item))

    def count(self, item):
        return super(WeakList, self).count(_wrap(item))

    def pop(self, index=-1):
        return _unwrap(super(WeakList, self).pop(index))

    def insert(self, index, item):
        return super(WeakList, self).insert(index, _wrap(item))

    def extend(self, items):
        return super(WeakList, self).extend(map(_wrap, items))

    def sort(self, *args, **kwargs):
        result = list(map(_unwrap, self)).sort(*args, **kwargs)
        self.clear()
        self.extend(result)
