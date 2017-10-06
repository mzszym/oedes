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

from .base import *
from oedes import functions


class BulkSource(object):
    def setUp(self, model):
        pass

    def evalute(self, vars):
        raise NotImplementedError()


class Interface(object):

    def __init__(self, equation, other_equation, name):
        self.equation, self.other_equation, self.name = equation, other_equation, name

    def setUp(self, model):
        self.mesh = self.equation.mesh
        self.other_mesh = self.other_equation.mesh
        self.boundary, self.other_boundary = self.mesh.getSharedBoundary(
            self.other_mesh, self.name)
        self.bidx, self.other_bidx = self.boundary[
            'bidx'], self.other_boundary['bidx']
        self.idx, self.other_idx = self.mesh.boundary.idx[
            self.bidx], self.other_mesh.boundary.idx[self.other_bidx]

    def evaluate(self, model, vars, vars_eq, other_vars_eq):
        raise NotImplementedError()

    def run(self, model, vars):
        eqvars = vars['eqvars']
        vars_eq = eqvars[id(self.equation)]
        other_vars_eq = eqvars[id(self.other_equation)]
        FdS, other_FdS, sources, other_sources = self.evaluate(
            model, vars, vars_eq, other_vars_eq)

        def add(dst, terms, bidx):
            for p, f in terms:
                dst[p].append((bidx, f))
        add(vars_eq['boundary_FdS'], FdS, self.bidx)
        add(other_vars_eq['boundary_FdS'], other_FdS, self.other_bidx)
        add(vars_eq['boundary_sources'], sources, self.bidx)
        add(other_vars_eq['boundary_sources'], other_sources, self.other_bidx)


class LangevinRecombination(BulkSource):
    "Langevinian recombination term, to be used"

    def __init__(self, mesh, electron_prefix='electron',
                 hole_prefix='hole', output_name='R'):
        self.mesh = mesh
        self.electron = electron_prefix
        self.hole = hole_prefix
        self.output_name = output_name

    @property
    def minus(self):
        return [self.electron, self.hole]

    @property
    def plus(self):
        return []

    def evaluate(self, vars):
        mu = vars['mu']
        c = vars['c']
        full_output = vars['full_output']
        R = functions.LangevinRecombination(mu[self.electron], mu[self.hole], c[self.electron], c[
            self.hole], self.mesh.cells['epsilon'], npi=vars['params']['npi'])
        if full_output is not None:
            assert self.output_name not in full_output
            full_output[self.output_name] = R
        return R


class SimpleGenerationTerm(BulkSource):
    def __init__(self, eq, function, prefix='absorption'):
        assert eq.z == 0
        self.function = function
        self.eq = eq
        self.prefix = self.eq.prefix + '.' + prefix
        self.G = self.function(self.eq.mesh.cells['center'])

    @property
    def plus(self):
        return [self.eq.prefix]

    @property
    def minus(self):
        return []

    def evaluate(self, vars):
        g = self.G * vars['params'][self.prefix + '.I']
        if vars['full_output'] is not None:
            vars['full_output'][self.prefix + '.G'] = g
        return g


class SimpleDecayTerm(BulkSource):
    def __init__(self, eq, prefix='decay'):
        assert eq.z == 0
        self.eq = eq
        self.prefix = self.eq.prefix + '.' + prefix

    @property
    def minus(self):
        return [self.eq.prefix]

    @property
    def plus(self):
        return []

    def evaluate(self, vars):
        return vars['c'][self.eq.prefix] * vars['params'][self.prefix]


class OnsagerBraunRecombinationDissociationTerm(BulkSource):
    b_eps = 1e-10
    b_max = 100

    def __init__(self, eq, electron_eq, hole_eq, prefix='dissociation'):
        assert eq.z == electron_eq.z + hole_eq.z, 'inconsistent signs'
        self.eq = eq
        self.electron_eq = electron_eq
        self.hole_eq = hole_eq
        self.prefix = self.eq.prefix + '.' + prefix

    @property
    def plus(self):
        return [self.eq.prefix]

    @property
    def minus(self):
        return [self.electron_eq.prefix, self.hole_eq.prefix]

    def evaluate(self, vars):
        mu = vars['mu']
        c = vars['c']
        params = vars['params']
        full_output = vars['full_output']
        a = params[self.eq.prefix + '.distance']
        u = 3. / (4. * np.pi * a**3)  # m^-3
        v = np.exp(-scipy.constants.elementary_charge /
                   (4 * np.pi * vars['epsilon'] * a * vars['Vt']))  # 1
        b = (scipy.constants.elementary_charge / (8 * np.pi)) * \
            vars['Ecellm'] / (vars['epsilon'] * vars['Vt']**2)  # 1
        # b=0.
        t = functions.OnsagerFunction(
            where(
                b < self.b_max,
                b,
                self.b_max) +
            self.b_eps)  # 1
        gamma = scipy.constants.elementary_charge * \
            (mu[self.electron_eq.prefix] + mu[self.hole_eq.prefix]) / \
            vars['epsilon']  # m^3 /s
        # print('\ngamma=%s,\nc=%s,u,v,t=%s\n,%s\n,%s\n'%(gamma.value,c[self.eq.prefix].value,u,v,t.value))
        r = gamma * (c[self.electron_eq.prefix] *
                     c[self.hole_eq.prefix] - params['npi'])  # 1/(m^3 s)
        d = gamma * c[self.eq.prefix] * u * v * t
        if full_output is not None:
            full_output[self.prefix + '.recombination'] = r
            full_output[self.prefix + '.dissociation'] = d
        return r - d


class TrapSource(BulkSource):
    "Source term fro conduction level to trap level"

    def __init__(self, transport_eq, trap_eq, fill_transport=True,
                 fill_trap=True, rrate_param=False):
        assert transport_eq.mesh is trap_eq.mesh, "the same mesh must be used for conduction and trap level"
        assert transport_eq.z == trap_eq.z, "inconsistent signs"

        self.transport_eq = transport_eq
        self.trap_eq = trap_eq
        self.fill_transport = fill_transport
        self.fill_trap = fill_trap
        self.rrate_param = rrate_param

    @property
    def minus(self):
        return [self.transport_eq.prefix]

    @property
    def plus(self):
        return [self.trap_eq.prefix]

    def evaluate(self, vars):
        params = vars['params']
        c = vars['c']
        transportlevel = params[self.transport_eq.prefix + '.level']
        trate = params[self.trap_eq.prefix + '.trate']
        if self.rrate_param:
            rrate = params[self.trap_eq.prefix + '.rrate']
        else:
            traplevel = params[self.trap_eq.prefix + '.level']
            depth = (transportlevel - traplevel) * self.transport_eq.z
            rrate = trate * exp(-depth / vars['Vt'])

        transportN0 = params[self.transport_eq.prefix + '.N0']
        trapN0 = params[self.trap_eq.prefix + '.N0']
        ctransport = c[self.transport_eq.prefix]
        ctrap = c[self.trap_eq.prefix]
        if self.fill_trap:
            a = trate * (trapN0 - ctrap)
        else:
            a = trate * trapN0
        if self.fill_transport:
            b = rrate * (transportN0 - ctransport)
        else:
            b = rrate * transportN0
        return a * ctransport - b * ctrap


class SRHRecombination(object):
    def __init__(self, transport_eq, trap_eq):
        assert transport_eq.z == -trap_eq.z, 'inconsistent signs'
        self.transport_eq = transport_eq
        self.trap_eq = trap_eq

    @property
    def plus(self):
        return []

    @property
    def minus(self):
        return [self.transport_eq.prefix, self.trap_eq.prefix]

    def evaluate(self, vars):
        mu = vars['mu']
        c = vars['c']
        return scipy.constants.elementary_charge * \
            mu[self.transport_eq.prefix] * c[self.transport_eq.prefix] * \
            c[self.trap_eq.prefix] / vars['epsilon']
