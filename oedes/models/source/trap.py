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

from .source import BulkSource
from oedes.ad import exp

__all__ = ['TrapSource']


class TrapSource(BulkSource):
    "Source term fro conduction level to trap level"

    def __init__(self, transport_eq, trap_eq, fill_transport=True,
                 fill_trap=True, rrate_param=False):
        super(
            TrapSource,
            self).__init__(
            'trap',
            eq_refs=[
                'transport_eq',
                'trap_eq'])
        assert transport_eq.mesh is trap_eq.mesh, "the same mesh must be used for conduction and trap level"
        assert transport_eq.z == trap_eq.z, "inconsistent signs"
        self.transport_eq = transport_eq
        self.trap_eq = trap_eq
        self.fill_transport = fill_transport
        self.fill_trap = fill_trap
        self.rrate_param = rrate_param

    def evaluate(self, ctx, eq):
        if ctx.solver.poissonOnly:
            return
        transportenergy = ctx.param(eq.transport_eq,  'energy')
        trate = ctx.param(eq.trap_eq, 'trate')
        if self.rrate_param:
            rrate = ctx.param(eq.trap_eq, 'rrate')
        else:
            trapenergy = ctx.param(eq.trap_eq, 'energy')
            depth = (trapenergy - transportenergy) * eq.transport_eq.z
            rrate = trate * \
                exp(-depth / ctx.varsOf(eq.transport_eq.thermal)['Vt'])
        transportN0 = ctx.param(eq.transport_eq, 'N0')
        trapN0 = ctx.param(eq.trap_eq, 'N0')
        ctransport = ctx.varsOf(eq.transport_eq)['c']
        ctrap = ctx.varsOf(eq.trap_eq)['c']
        if self.fill_trap:
            a = trate * (trapN0 - ctrap)
        else:
            a = trate * trapN0
        if self.fill_transport:
            b = rrate * (transportN0 - ctransport)
        else:
            b = rrate * transportN0
        g = a * ctransport - b * ctrap
        self.add(ctx, g, plus=[eq.trap_eq], minus=[eq.transport_eq])
