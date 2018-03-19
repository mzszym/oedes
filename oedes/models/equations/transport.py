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

from .base import ConservationEquation, DelegateConvergenceTest, Funcall
from oedes.fvm import ElementwiseConvergenceTest
from .poisson import Poisson
import numpy as np
from oedes.ad import dot, sparsesum, getitem, nvalue
from oedes.models import solver
import weakref

__all__ = ['Transport', 'TransportCharged']


class Transport(ConservationEquation):
    def __init__(self, *args, **kwargs):
        super(Transport, self).__init__(*args, **kwargs)
        self.convergenceTest = ElementwiseConvergenceTest(
            atol=5e7, rtol=1e-7, eps=1e-10)
        self.defaultBCConvergenceTest = ElementwiseConvergenceTest(
            atol=1., rtol=1e-7)

        self.poisson = None
        self.thermal = None
        self.v_D = None
        self.dos = None

    def build(self, builder):
        obj = builder.newTransporteEquation(
            builder.getMesh(self.mesh), self.name)
        self._build_transport(builder, obj)
        return obj

    def _build_transport(self, builder, obj):
        obj.convergenceTest = DelegateConvergenceTest(self)
        assert isinstance(self.poisson, Poisson)
        builder.addInitializer(self._init, obj)
        obj.load = Funcall(self.load, obj)
        obj.evaluate = Funcall(self.evaluate, obj, depends=[obj.load])
        builder.addEvaluation(obj.load)
        builder.addEvaluation(obj.evaluate)
        obj.prefix = self.prefix
        builder.add(self, obj)

    def _init(self, builder, obj):
        obj.poisson = builder.get(self.poisson)
        obj.thermal = builder.get(self.thermal)
        obj.poisson.species.append(obj)
        obj.poisson.evaluate.depends(obj.load)
        obj.poisson.allspecies.depends(obj.evaluate)
        obj.load.depends(obj.poisson.load)
        obj.load.depends(obj.thermal.load)

    def load(self, ctx, eq):
        super(Transport, self).load(ctx, eq)
        newvars = ctx.varsOf(eq)
        newvars['c'] = newvars['x']
        newvars['ct'] = newvars['xt']
        if ctx.solver.poissonOnly:
            return
        newvars['sources'] = 0
        newvars['boundary_sources'] = []
        newvars['boundary_FdS'] = []
        assert eq.poisson.mesh is eq.mesh, 'multiple meshes currently unsupported'
        if self.dos is not None:
            Ef = self.dos.QuasiFermiLevel(ctx, eq)
            newvars['Ef'] = Ef
        newvars['v_D'] = self.v_D(ctx, eq)

    def evaluate(self, ctx, eq):
        newvars = ctx.varsOf(eq)
        if ctx.solver.poissonOnly:
            yield eq.identity(newvars['c'])
            return
        sources = newvars['sources']
        c = newvars['c']
        ct = newvars['ct']
        v_D = newvars['v_D']
        ctx.outputCell([eq, 'c'], c, unit=ctx.units.concentration)
        ctx.output([eq, 'ct'], ct, unit=ctx.units.dconcentration_dt)
        if v_D is None:
            j = np.zeros(len(eq.mesh.faces))
            newvars['j'] = None
        else:
            v, D = v_D
            f = eq.faceflux(c=c, v=v, D=D, full_output=ctx.wants_output)
            if ctx.wants_output:
                j = f['flux']
                ctx.outputFace([eq, 'j'], f['flux'], unit=ctx.units.flux)
                ctx.outputFace([eq, 'jdrift'], f['flux_v'],
                               unit=ctx.units.flux)
                ctx.outputFace([eq, 'jdiff'], f['flux_D'], unit=ctx.units.flux)
                if self.dos is not None:
                    ctx.outputCell([eq, 'Ef'], newvars['Ef'],
                                   unit=ctx.units.eV)
            else:
                j = f
            newvars['j'] = j
        yield eq.residuals(eq.mesh.internal, j, celltransient=ct, cellsource=sources)
        FdS = dot(eq.mesh.boundary.fluxsum, j) + \
            sparsesum(len(eq.mesh.boundary.cells),
                      newvars['boundary_FdS'])
        newvars['J_boundary'] = FdS * eq.ze
        boundary_sources = sparsesum(len(eq.mesh.boundary.cells), newvars[
                                     'boundary_sources'])
        self.evaluate_bc(ctx, eq, c, j, FdS, celltransient_boundary=ct[
            eq.mesh.boundary.idx], cellsource_boundary=getitem(sources, eq.mesh.boundary.idx) + boundary_sources)


class TransportCharged(Transport):
    def __init__(self, mesh, name, z):
        super(TransportCharged, self).__init__(mesh, name)
        self.z = z

    def build(self, builder):
        obj = builder.newTransportChargedEquation(
            builder.getMesh(self.mesh), self.name, self.z)
        self._build_transport(builder, obj)
        return obj
