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

import collections
from .computation import Computation

__all__ = ['OutputMeta', 'Units', 'FreedVars', 'EvaluationContext', 'meta_key']

OutputMeta = collections.namedtuple('OutputMeta', ['mesh', 'face', 'unit'])


class Units:
    V = 'V'
    potential = V
    charge_density = 'C/m^3'
    electric_field = 'V/m'
    current_density = 'A/m^2'
    concentration = '1/m^3'
    dconcentration_dt = '1/(m^3 s)'
    flux = '1/(m^3 s)'
    eV = 'eV'
    meter = 'm'

    @classmethod
    def check(cls, unit):
        assert unit in [
            None,
            cls.V,
            cls.charge_density,
            cls.electric_field,
            cls.current_density,
            cls.concentration,
            cls.flux,
            cls.eV,
            cls.dconcentration_dt]


meta_key = '.meta'


class FreedVars(object):
    pass


class EvaluationContext(object):
    UPPER = '..'

    units = Units

    def __init__(self, params, outputs, vardict=None, gvars=None):
        if vardict is None:
            vardict = dict()
        assert gvars is not None
        self._vardict = vardict
        self.params = params
        self.outputs = outputs
        self.gvars = gvars

        # TODO:
        if self.outputs is not None:
            if meta_key not in self.outputs:
                self.outputs[meta_key] = dict()
            self.outputs_meta = self.outputs[meta_key]

    def newVars(self, obj, varobj=None):
        if varobj is None:
            varobj = dict()
        assert not isinstance(obj, Computation)
        k = id(obj)
        assert k not in self._vardict
        self._vardict[k] = varobj
        return varobj

    def varsOf(self, obj):
        assert not isinstance(obj, Computation)
        return self._vardict[id(obj)]

    def killVars(self, obj):
        assert not isinstance(obj, Computation)
        self._vardict[id(obj)] = FreedVars()

    @property
    def wants_output(self):
        return self.outputs is not None

    def _path(self, eq, *args):
        if eq is not None and eq.prefix:
            path = list(eq.prefix.split('.'))
        else:
            path = []
        for a in args:
            if a is self.UPPER:
                path = path[:-1]
            else:
                path.append(a)
        return '.'.join(path)

    def _absdict(self, obj):
        return obj

    def param(self, *args):
        p = self._path(*args)
        return self._absdict(self.params)[p]

    def output(self, args, value, mesh=None, face=False, unit=None, add=False):
        if self.outputs is None:
            return
        p = self._path(*args)
        d = self._absdict(self.outputs)
        meta = self._absdict(self.outputs_meta)
        if add and p in d:
            d[p] = d[p] + value
        else:
            assert p not in d, 'duplicate assignment to %r' % p
            d[p] = value
        Units.check(unit)
        meta_info = OutputMeta(mesh=mesh, face=face, unit=unit)
        if add and p in meta:
            assert meta[p] == meta_info
        else:
            meta[p] = meta_info

    def common_param(self, eqs, *args):
        p = self._path(eqs[0], self.UPPER, *args)
        assert all([self._path(e, self.UPPER, *args) == p for e in eqs[1:]])
        return self._absdict(self.params)[p]

    def _meshOutput(self, args, value, mesh=None, unit=None, face=None):
        if mesh is None:
            mesh = args[0].mesh
        return self.output(args, value, mesh=mesh, face=face, unit=unit)

    def outputFace(self, args, value, **kwargs):
        return self._meshOutput(args, value, face=True, **kwargs)

    def outputCell(self, args, value, **kwargs):
        return self._meshOutput(args, value, face=False, **kwargs)

    def __getattr__(self, name):
        if name in self.gvars:
            return self.gvars[name]
        else:
            return super(EvaluationContext, self).__getattr__(self, name)
