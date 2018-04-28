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

# Basic equations
#
# Using vertex centered finite volume method to discretize the equations.
# $\frac{du}{dt} = S + \nabla \cdot F $
# by discretized applying Gauss theorem
# $\int_V \frac{du}{dt}dV = \int_V S dV + \int_\Omega F d\Omega$
# where $V$ denotes cell volume, and $\Omega$ denotes cell boundary.
#
# In discretized geometry
# $V \frac{dc_i}{dt} = V s_i + \sum_k A_{ik} f_{k}$
# This is conveniently expressed by Equation.conservation
# $\frac{d\mathbf{c}}{dt} - \mathbf{s} - V^-1 \mathbf{A} \mathbf{f}=F_{internal}$
# where $V^-1$ is diagonal matrix of cell volumes. A is called fluxsum in the code.
#
# At boundary, values calculated by above equation are replaced by
# $b(V^-1 \mathbf{A} \mathbf{f},...)=F_{boundary}$
#
# Final residuals are given as $F_{internal}$ for internal points, and
# $F_{boundary}$ for boundary cells.

from oedes import ad
import numpy as np
from .discrete import *


def _boundaryIdx(eq, boundary):
    return eq.mesh.boundary.idx[boundary['bidx']]


def _boundaryDof(eq, boundary):
    return eq.idx[_boundaryIdx(eq, boundary)]


class FVMBoundaryEquation(DiscreteEquation):
    """
    Cell-center finite volume boundary equation

    Members:
    ---------------------------
    owner_eq : FVMConservationEquation
        Equation this boundary condition applies to
    other_eq_lookup : callable(FVMBuilder) -> FVMConservationEquation
        For internal interfaces: find equation on the other side of interface
    name : str
        Name of this interface
    other_eq : FVMConservationEquation
        For internal interface, after setUp : equation on the other side of the interface
    boundary :
        result of mesh.getBoundary or mesh.getSharedBoundary
    other_boundary :
        For internal interfaces: result of mesh.getSharedBoundary, applies to the other side of the interface
    conservation_to_dof : integer array
        For internal interfaces: destination cell for conservation
    """

    def __init__(self, owner_eq, name, other_eq_lookup=None):
        super(FVMBoundaryEquation, self).__init__()
        self.owner_eq = owner_eq
        self.other_eq_lookup = other_eq_lookup
        self.name = name

    def _getDof(self):
        return _boundaryDof(self.owner_eq, self.boundary)

    def _getIdx(self):
        return _boundaryIdx(self.owner_eq, self.boundary)

    def _getOtherIdx(self):
        return _boundaryIdx(self.other_eq, self.other_boundary)

    def init(self, builder):
        if self.other_eq_lookup is None:
            self.other_eq = None
            self.boundary = self.owner_eq.mesh.getBoundary(self.name)
            self.other_boundary = None
            self.conservation_to_dof = _boundaryDof(
                self.owner_eq, self.boundary)
        else:
            self.other_eq = self.other_eq_lookup(builder)
            self.boundary, self.other_boundary = self.owner_eq.mesh.getSharedBoundary(
                self.other_eq.mesh, self.name)
            self.conservation_to_dof = _boundaryDof(
                self.other_eq, self.other_boundary)
        assert self.owner_eq.bc_dof_is_free[self.boundary['bidx']].all(
        ), 'not overriding BCs'
        self.owner_eq.bc_dof_is_free[self.boundary['bidx']] = False

    def ndof(self):
        return 0

    def testConverged(self, F, x, dx, report):
        return self.convergenceTest.testBoundary(
            self.owner_eq, self, F, x, dx, report)


BoundaryEquation = FVMBoundaryEquation


class FVMConservationEquation(DiscreteEquation):
    """
    Basic equation defining one degree of freedom per cell, solved using vertex centered finite volume

    Properties
    ----------

    transientvar : float
        one if pseudotransient techniques should be applied to this equation, zero if equation is time-independent
    idx : array of int
        indices of degrees of freedom in solution vector corresponding to this equation
    convergenceTest : ConvergenceTest
        convergence criteria for this equation
    prefix : str
        prefix of this equation used for parameters and output
    boundary_labels : integer array
        For internal interfaces: labels of merged interfacial cells
    """

    def __init__(self, mesh, name):
        """Abstract block of equation, which introduces a field into equation system on :mesh:
        Keeps track of:
        indices of belonging to the field :idx:, x[idx[i]] is field in cell i
        :name:, which currently is the same as prefix
        :boundary: cells
        """
        super(FVMConservationEquation, self).__init__()
        self.mesh = mesh
        self.name = name
        self.bc = []

        self.bc_dof_is_free = np.ones_like(mesh.boundary.idx, dtype=np.bool)
        self.boundary_labels = None

    def residuals(self, part, facefluxes, celltransient=0., cellsource=0.):
        """
        Calculate FVM residuals

        Parameters:
        -----------
        facefluxes: vector or None
            fluxes per unit of surface or None
        celltransient: float or vector
            transient term per unit of volume
        cellsource: float or vector
            source term per unit of volume


        Returns: tuple (residuals,FdS)
        --------
        residuals:
            residuals for each cell, per unit of volume
        FdS:
            integral FdS per cell
        """

        assert part is self.mesh.internal or part is self.mesh.boundary
        idx = part.idx
        if facefluxes is None:
            FdS = 0.
        else:
            FdS = ad.dot(part.fluxsum, facefluxes)
        if not ad.isscalar(celltransient):
            celltransient = celltransient[idx]
        if not ad.isscalar(cellsource):
            cellsource = cellsource[idx]
        return self.idx[idx], -FdS / \
            part.cells['volume'] + celltransient - cellsource

    def ndof(self):
        return self.mesh.ncells

    def identity(self, x):
        return self.idx, x - ad.nvalue(x)
