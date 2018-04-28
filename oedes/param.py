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

class paramvector_params():
    def __init__(self, base, values, ix):
        self.base = base
        self.values = values
        self.ix = ix

    def __getitem__(self, u):
        if u in self.ix:
            return self.values[self.ix[u]]
        else:
            return self.base[u]


class paramvector():
    def __init__(self, names):
        assert len(set(names)) == len(names), 'do not duplicate'
        self.ordering = tuple(names)
        self.ix = dict((name, i) for (i, name) in enumerate(self.ordering))

    def values(self, params):
        return [params[name] for i, name in enumerate(self.ordering)]

    def asdict(self, base, values):
        return paramvector_params(base, values, self.ix)
