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

from oedes import model
from oedes.ad import *
from oedes.fvm import BoundaryEquation, ElementwiseConvergenceTest
from oedes.fvm.poisson import Poisson
from oedes.fvm.transport import Transport, TransportCharged
from oedes.util import TODOWarning
from .compose import CompositeModel
import numpy as np
import scipy.constants
import itertools
import collections
import warnings


class BaseModel(CompositeModel):
    """
    Basic device model:
    Supports arbitrary number of species (transported or not) coupled to Poisson's equation, with any form of
    boundary conditions and drift and diffusion velocities
    """

    poisson = None
    species = None
    species_dos = None
    species_v_D = None
    sources = None
    other_equations = None
    additional_charge = None
    ordering = None
    after_load = None

    def __init__(self, *args, **kwargs):
        CompositeModel.__init__(self)
        self.species = []
        self.sources = []
        self.species_v_D = dict()
        self.species_dos = dict()
        self.other_equations = []
        self.additional_charge = []
        self.ordering = 'cell'

    def init_equations(self):
        assert isinstance(self.poisson, Poisson)
        assert len(set([eq.prefix for eq in self.all_equations])) == len(
            list(self.all_equations)), "duplicate prefix of equation"
        if self.poisson.convergenceTest is None:
            self.poisson.convergenceTest = ElementwiseConvergenceTest(
                atol=1e-15, rtol=1e-7)
            for bc in self.poisson.bc:
                if bc.convergenceTest is None:
                    bc.convergenceTest = ElementwiseConvergenceTest(
                        atol=1e-15, rtol=1e-7)
        for eq in self.species:
            assert isinstance(eq, Transport)
            if eq.convergenceTest is None:
                eq.convergenceTest = ElementwiseConvergenceTest(
                    atol=5e7, rtol=1e-7)
            for bc in eq.bc:
                if bc.convergenceTest is None:
                    bc.convergenceTest = ElementwiseConvergenceTest(
                        atol=1., rtol=1e-7)
        for eq in self.all_equations:
            assert eq.mesh is self.poisson.mesh, "multiple meshes not supported"

        CompositeModel.init_equations(self)

    def create_indexes(self):
        assert self.ordering in ['cell', 'equation']
        if self.ordering == 'equation':
            i = 0
            for eq in self.all_equations:
                assert eq.idx is None, "already indexed"
                ndof = eq.mesh.ncells
                eq.idx = np.arange(i, i + ndof, dtype=np.int)
                i += ndof
            self.X = np.zeros(i)
        else:
            neq = len(list(self.all_equations))
            for i, eq in enumerate(self.all_equations):
                assert eq.idx is None, "already indexed"
                assert eq.mesh is self.poisson.mesh, "multiple meshes not supported"
                eq.idx = np.arange(0, eq.mesh.ncells) * neq + i
            self.X = np.zeros(neq * self.poisson.mesh.ncells)

    @property
    def equations(self):
        return itertools.chain(
            [self.poisson], self.species, self.other_equations)

    def update_equations(self, x, params):
        # warnings.warn(
        #    'clipping to N0 also for ions, if specied in parameters : this could be very wrong!', TODOWarning)
        for eq in self.species:
            xe = x[eq.idx]
            x[eq.idx] = np.where(xe < 0., 0., xe)
            # if eq.prefix+'.N0' in params:
            #    x[eq.idx]=np.clip(xe,0.,params[eq.prefix+'.N0'])
            # else:
            #    x[eq.idx]=np.clip(xe,0.,xe)

    def evaluate_bc(self, eq, v, fluxes, FdS_boundary, vars,
                    celltransient_boundary=0., cellsource_boundary=0.):
        bclist = eq.bc
        conservation = -FdS_boundary + eq.mesh.cells['volume'][
            eq.mesh.boundary.idx] * (celltransient_boundary - cellsource_boundary)
        vars['bc_conservation'].append((eq.boundary_labels, conservation))
        for bc in bclist:
            yield eq.idx[eq.mesh.boundary.idx[bc.bidx]], bc.residuals(eq, v, FdS_boundary, vars)

    def evaluate_poisson(self, vars):
        "Part of evaluate dealing with Poisson's equation"
        c, params, E, Et, full_output = vars['c'], vars[
            'params'], vars['E'], vars['Et'], vars['full_output']
        epsilon = vars['epsilon']
        species_charge = sum([eq.ze * c[eq.prefix] for eq in self.species])
        additional_charge = sum([f(vars) for f in self.additional_charge])
        total_charge_density = species_charge + additional_charge
        faceepsilon = self.poisson.mesh.faceaverage(epsilon)
        D = self.poisson.displacement(E, faceepsilon)
        Dt = self.poisson.displacement(Et, faceepsilon)
        if full_output is not None:
            full_output['poisson.total_charge_density'] = total_charge_density
            full_output['potential'] = vars['potential']
            full_output['E'] = E
            full_output['Et'] = Et
            full_output['D'] = D
            full_output['Dt'] = Dt
        yield self.poisson.residuals(self.poisson.mesh.internal, D, cellsource=total_charge_density)
        FdS_boundary = dot(self.poisson.mesh.boundary.fluxsum, D)
        vars['Jd_boundary'] = dot(self.poisson.mesh.boundary.fluxsum, Dt)
        for res in self.evaluate_bc(self.poisson, vars[
                                    'potential'], D, FdS_boundary, vars):
            yield res

    def evaluate_source_terms(self, vars):
        "Part of evaluate dealing with source terms"

        sources = dict([(eq.prefix, 0.) for eq in self.species])
        for src in self.sources:
            f = src.evaluate(vars)
            for p in src.plus:
                sources[p] = sources[p] + f
            for n in src.minus:
                sources[n] = sources[n] - f
        return sources

    def evaluate_species(self, vars):
        "Part of evaluate dealing with residuals of species"

        E, sources, out = vars['E'], vars['sources'], vars['full_output']
        want_output = out is not None
        if out is not None:
            J = sum(out['Dt'] * self.poisson.mesh.faces['dr']) / \
                sum(self.poisson.mesh.faces['dr'])
        for eq in self.species:
            c = vars['c'][eq.prefix]
            ct = vars['ct'][eq.prefix]
            v_D = vars['v_D'][eq.prefix]
            if want_output:
                out[eq.prefix + '.c'] = c
                out[eq.prefix + '.ct'] = ct
            if v_D is None:
                j = np.zeros(len(eq.mesh.faces))
            else:
                v, D = v_D
                f = eq.faceflux(c=c, v=v, D=D, full_output=want_output)
                if want_output:
                    j = f['flux']
                    out[eq.prefix + '.j'] = f['flux']
                    out[eq.prefix + '.jdrift'] = f['flux_v']
                    out[eq.prefix + '.jdiff'] = f['flux_D']
                    if eq.prefix in vars['Ef']:
                        out[eq.prefix + '.Ef'] = vars['Ef'][eq.prefix]
                    J = J + sum(j * eq.ze *
                                eq.mesh.faces['dr']) / sum(eq.mesh.faces['dr'])
                else:
                    j = f
            yield eq.residuals(eq.mesh.internal, j, celltransient=ct, cellsource=sources[eq.prefix])
            FdS = dot(eq.mesh.boundary.fluxsum, j) + \
                sparsesum(len(eq.mesh.boundary.cells),
                          vars['boundary_FdS'][eq.prefix])
            vars['J_boundary'][eq.prefix] = FdS * eq.ze
            boundary_sources = sparsesum(len(eq.mesh.boundary.cells), vars[
                                         'boundary_sources'][eq.prefix])

            for res in self.evaluate_bc(eq, c, j, FdS, vars, celltransient_boundary=ct[
                                        eq.mesh.boundary.idx], cellsource_boundary=getitem(sources[eq.prefix], eq.mesh.boundary.idx) + boundary_sources):
                yield res
        if want_output:
            out['J'] = J

    # The design of this routine tries to reduce the cost of automatic differentation:
    # 1) Parts of x, xt corresponding to each field are taken once and put in c,ct etc.
    #    They should not be taken again in calculation routines.
    # 2) All calculation routines get common loc dict, which allows to store once
    #    calculated variables for reuse.
    # 3) The order of evaluation is as follows:
    #    - Poisson's equation
    #    - Species velocities v and diffusion coefficients D
    #      This usually involves calculation of mobilities which are not used
    #      directly by flux conservation, but are normally stored for sources.
    #    - Sources
    #    - Conservation of species (using sources, v and D)
    def load_equations(self, loc):
        x, xt, params = loc['x'], loc['xt'], loc['params']
        T = params['T']
        Vt = params['T'] * scipy.constants.Boltzmann / \
            scipy.constants.elementary_charge
        potential = x[self.poisson.idx]
        E = self.poisson.E(potential)
        Et = self.poisson.E(xt[self.poisson.idx])
        epsilon = scipy.constants.epsilon_0 * params['epsilon_r']
        Ecellv = self.poisson.mesh.cellaveragev(E)
        Ecellm = self.poisson.mesh.magnitudev(Ecellv)
        c = dict([(eq.prefix, x[eq.idx]) for eq in self.species])
        ct = dict([(eq.prefix, xt[eq.idx]) for eq in self.species])
        loc.update(dict(T=T, Vt=Vt, potential=potential, E=E, Et=Et, epsilon=epsilon, Ecellv=Ecellv, Ecellm=Ecellm,
                        c=c, ct=ct, FdS_boundary={}, J_boundary={}, Ef={}, dos={}, dos_data={}))
        for eq in self.species:
            if eq.prefix in self.species_dos:
                dos = self.species_dos[eq.prefix]
                loc['dos'][eq.prefix] = dos
                loc['dos_data'][eq.prefix] = dict()
                Ef = dos.QuasiFermiLevel(eq, loc)
                loc['Ef'][eq.prefix] = Ef
        loc['mu'] = dict()
        loc['v_D'] = dict([(eq.prefix, self.species_v_D[eq.prefix](eq, loc))
                           for eq in self.species if eq.prefix])
        loc['sources'] = self.evaluate_source_terms(loc)
        loc['boundary_FdS'] = dict([(eq.prefix, []) for eq in self.species])
        loc['boundary_sources'] = dict(
            [(eq.prefix, []) for eq in self.species])

    def generate_residuals_equations(self, loc):
        for res in self.evaluate_poisson(loc):
            yield res
        for res in self.evaluate_species(loc):
            yield res
        # for res in self.generate_bc_conservation(loc):
        #    yield res
