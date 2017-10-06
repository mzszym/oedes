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


class ConvergenceTest(object):
    pass


class NormConvergenceTest(ConvergenceTest):

    def __init__(self, stol=1e-6, atol=0.):
        self.stol = stol
        self.atol = atol

    def testEquation(self, eq, F, x, dx, params):
        F = F[eq.idx]
        x = x[eq.idx]
        dx = dx[eq.idx]
        if np.linalg.norm(dx) < self.stol * np.linalg.norm(x):
            return True
        if np.linalg.norm(F) < self.atol * np.sqrt(len(F)):
            return True
        return False

    def testBoundary(self, eq, bc, F, x, dx, params):
        raise NotImplementedError()


class ElementwiseConvergenceTest(ConvergenceTest):

    def __init__(self, rtol=1e-6, atol=0., drtol=1e-6, datol=0.):
        self.rtol = rtol
        self.atol = atol

    def _testix(self, eq, ix, F, x, dx, report, prefix):
        i = eq.idx[ix]
        if len(i) == 0:
            return True
        F = F[i]
        x = x[i]
        dx = dx[i]
        converged_abs = np.abs(F) < self.atol
        converged_rel = np.abs(dx) < self.rtol * np.abs(x)
        converged = np.logical_or(converged_abs, converged_rel).all()

        def amax(v):
            if len(v) == 0:
                return 0.
            else:
                return np.amax(v)
        with np.errstate(divide='ignore', invalid='ignore'):
            report[prefix] = dict(converged=converged,
                                  atol=self.atol,
                                  rtol=self.rtol,
                                  maxabs=amax(np.abs(F)),
                                  maxabs_nc=amax(
                                      np.abs(F[np.logical_not(converged_rel)])),
                                  maxrel=amax(np.abs(dx / x)),
                                  maxrel_nc=amax(np.abs((dx / x)[np.logical_not(converged_abs)])))
        return converged

    def testEquation(self, eq, F, x, dx, params, report):
        ix_all = np.hstack(
            [eq.mesh.internal.idx, eq.mesh.boundary.idx[eq.bc_free_dof]])
        return self._testix(eq, ix_all, F, x, dx, report, eq.prefix)

    def testBoundary(self, eq, bc, F, x, dx, params, report):
        return self._testix(eq, eq.mesh.boundary.idx[
                            bc.bidx], F, x, dx, report, eq.prefix + '.' + bc.name)


class BoundaryEquation(object):
    idx = None
    mesh = None
    bidx = None
    conservation_to = None

    convergenceTest = None

    def setUp(self, eq):
        assert self.idx is None and self.mesh is None, 'object already setUp'
        self.idx = eq.idx
        self.mesh = eq.mesh
        self.conservation_to = self.idx[
            self.mesh.boundary.idx[self.bidx]].copy()

    def conservation(self, eq, FdS, celltransient=0., cellsource=0.):
        if not ad.isscalar(celltransient):
            celltransient = celltransient[self.bidx]
        if not ad.isscalar(cellsource):
            cellsource = cellsource[self.bidx]
        volume = eq.boundary.cells['volume'][self.bidx]
        return -FdS + volume * (celltransient - cellsource)


class CellEquation(object):
    convergenceTest = None
    idx = None
    bc = None
    bc_free_dof = None
    bc_conservation_labels = None

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
    """

    def __init__(self, mesh, name, boundary=None):
        """Abstract block of equation, which introduces a field into equation system on :mesh:
        Keeps track of:
        indices of belonging to the field :idx:, x[idx[i]] is field in cell i
        :name:, which currently is the same as prefix
        :boundary: cells
        """
        self.mesh = mesh
        self.name = name
        self.bc = []

    @property
    def prefix(self):
        return self.name

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

    @property
    def transientvar(self):
        return 1.

    def testConverged(self, F, x, dx, params, report):
        tests = [self.convergenceTest.testEquation(self, F, x, dx, params, report)] + [
            bc.convergenceTest.testBoundary(self, bc, F, x, dx, params, report) for bc in self.bc]
        return all(tests)
