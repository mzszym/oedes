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

import numpy as np
import scipy.sparse
import scipy.constants
from oedes.ad import isscalar, stack, dot
from collections import OrderedDict


class meshpart(object):
    "Part of mesh"
    cells = None
    idx = None
    fluxsum = None

    def __init__(self, idx=None):
        self.idx = idx
        self.default_electrode = None

    def setUp(self):
        pass

    @property
    def ncells(self):
        return len(self.cells)

    @property
    def x(self):
        return self.cells['center']


class mesh(meshpart):
    "Vertex centered finite volume mesh"
    cells = None
    faces = None
    fluxsum = None

    internal = None
    boundary = None

    group = None

    def getSharedBoundary(self, other, name):
        assert self.group is not None, 'must belong to group'
        return self.group.getSharedBoundary(self, other, name)

    def __init__(self):
        meshpart.__init__(self)

    def facegrad(self, x):
        return (x[self.faces['+']] - x[self.faces['-']]) / self.faces['dr']

    def faceaverage(self, x):
        if isscalar(x):
            return x
        else:
            return (x[self.faces['-']] + x[self.faces['+']]) * 0.5

    def cellaveragev(self, x):
        # unimplemented
        return np.NaN

    def magnitudev(self, x):
        # unimplemented
        return np.NaN

    def dotv(self, x, y):
        # unimplemented
        return np.NaN

    def setUp(self):
        _internal = np.ones(self.ncells, dtype=np.bool)
        _internal[self.boundary.idx] = False
        _internal_idx = np.arange(self.ncells)[_internal]
        self.internal = meshpart(_internal_idx)
        assert (np.unique(self.boundary.idx) == self.boundary.idx).all(
        ), "bad boundary subset : idx must be sorted and unique"
        assert len(self.internal.idx) + len(
            self.boundary.idx) == self.ncells, "bad boundary specification : duplicate idx"
        for obj in [self.internal, self.boundary]:
            obj.fluxsum = self.fluxsum[obj.idx]
            obj.cells = self.cells[obj.idx]
            obj.setUp()

    @property
    def ndim(self):
        c = self.cells['center']
        if len(c.shape) == 1:
            return 1
        else:
            return c.shape[1]


class mesh1d_group(object):

    def getSharedBoundary(self, mesh, other, name):
        assert other.group is self and mesh.group is self
        assert name in mesh.boundaries, 'boundary not known'
        other, = (k for k in mesh.boundaries if k != name)
        return mesh.boundaries[name], mesh.boundaries[other]


class mesh1d(mesh):
    "Unidimensional mesh"

    cellavg = None

    def spacing(self, L, n_desired=200, dx_boundary=1e-10):
        dx_dl = dx_boundary
        dx_desired = L / n_desired

        # Spacing for double layer region
        dldx_ = 10**np.linspace(np.log10(dx_dl), np.log10(dx_desired),
                                int(np.ceil(np.log10(dx_desired / dx_dl) * 10)))
        xdl_ = stack(0., np.cumsum(dldx_))
        n_normal_ = (L - (2 * xdl_[-1])) / dx_desired
        assert n_normal_ > 100
        xnorm_ = np.linspace(xdl_[-1], L - xdl_[-1], int(n_normal_))
        x_ = stack(xdl_, xnorm_[1:-1], L - xdl_[::-1])
        return x_

    def __init__(self, L, epsilon_r=3., boundary_names=[
                 'electrode0', 'electrode1'], **kwargs):
        mesh.__init__(self)

        if np.asarray(L).shape == ():
            x_ = self.spacing(L, **kwargs)
        else:
            x_ = L
            L = x_[-1]

        n = len(x_)

        # create cells
        cells = np.zeros(n, dtype=[('center', '1f8'),
                                   ('volume', 'f8'),
                                   ('epsilon', 'f8')])
        # cells['center']=np.linspace(0,L,n)
        cells['center'] = x_
        cells['epsilon'] = scipy.constants.epsilon_0 * epsilon_r
        _dx = np.diff(cells['center']) / 2
        cells['volume'] = np.hstack((_dx, 0)) + np.hstack((0, _dx))
        assert np.allclose(sum(cells['volume']), L)
        self.cells = cells
        assert np.allclose(sum(cells['volume']), L)

        # create faces
        faces = np.zeros(n - 1, dtype=[('-', 'i4'), ('+', 'i4'),
                                       ('surface', 'f8'),
                                       ('dR', '1f8'),
                                       ('dr', 'f8'),
                                       ('epsilon', 'f8'),
                                       ('center', '1f8')])
        faces['-'] = np.arange(0, n - 1)
        faces['+'] = np.arange(1, n)
        faces['surface'] = 1.
        faces['epsilon'] = 0.5 * \
            (cells['epsilon'][faces['-']] + cells['epsilon'][faces['+']])
        faces['center'] = 0.5 * \
            (cells['center'][faces['-']] + cells['center'][faces['+']])
        faces['dR'] = cells['center'][faces['+']] - cells['center'][faces['-']]
        faces['dr'] = np.sqrt(np.sum(np.atleast_2d(faces['dR']**2), axis=0))
        assert np.allclose(sum(faces['dr']), L)
        self.faces = faces

        # create boundary
        self.boundary = meshpart(idx=np.asarray([0, n - 1]))

        self.electrode0 = np.zeros(
            1, dtype=[('bidx', 'i4'), ('normal', '1f8'), ('surface', 'f8')])
        self.electrode1 = np.zeros(
            1, dtype=[('bidx', 'i4'), ('normal', '1f8'), ('surface', 'f8')])
        self.electrode0['bidx'] = 0
        self.electrode1['bidx'] = 1
        self.electrode0['normal'] = 1.
        self.electrode1['normal'] = -1.
        self.electrode0['surface'] = 1.
        self.electrode1['surface'] = 1.
        lname, rname = boundary_names
        self.boundaries = {lname: self.electrode0, rname: self.electrode1}
        self.group = mesh1d_group()

        # create flux summing matrix
        def _fluxsum(cells, fluxes):
            i = np.arange(len(fluxes))
            shape = (cells.shape[0], fluxes.shape[0])

            def m(sign):
                return scipy.sparse.csr_matrix(
                    (fluxes['surface'], (fluxes[sign], i)), shape=shape)
            return m('+') - m('-')

        self.fluxsum = _fluxsum(cells, faces)
        self.fluxsum.sort_indices()

        # create flux averaging matrix
        def _cellsum(fluxsum):
            a = fluxsum.tocoo()
            i, j = a.row, a.col
            A = scipy.sparse.csr_matrix(
                (np.ones(len(i)), (i, j)), shape=(len(self.cells), len(self.faces)))
            s = A.sum(axis=1).A1
            sc = scipy.sparse.diags(
                [1. / s], offsets=[0], shape=(len(self.cells), len(self.cells)))
            return (sc.dot(A)).tocsr()
        self.cellavg = _cellsum(self.fluxsum)
        self.default_electrode = boundary_names[-1]
        self.setUp()

    @property
    def length(self):
        return self.cells['center'][-1]

    def hasBoundary(self, name):
        return name in self.boundaries

    def getBoundary(self, name):
        return self.boundaries[name]

    def cellaveragev(self, x):
        if isscalar(x):
            return x
        return dot(self.cellavg, x)

    def magnitudev(self, x):
        return np.abs(x)

    def dotv(self, x, y):
        return x * y


def emptyBoundary():
    return np.zeros(
        0, dtype=[('bidx', 'i4'), ('normal', '1f8'), ('surface', 'f8')])


class multilayer1d_group:

    def getSharedBoundary(self, mesh, other, name):
        assert other.group is self and mesh.group is self
        assert name in mesh.boundaries, 'boundary not known'
        if other.layer_i == mesh.layer_i + \
                1 and mesh.boundaries[name] is mesh.electrode1:
            return (mesh.electrode1, other.electrode0)
        if other.layer_i == mesh.layer_i - \
                1 and mesh.boundaries[name] is mesh.electrode0:
            return (mesh.electrode0, other.electrode1)
        return None


class multilayer1d:

    def __init__(self, meshes):
        x = 0.
        self.domains = OrderedDict()
        self.group = multilayer1d_group()
        for i, (name, mesh) in enumerate(meshes):
            assert mesh.cells['center'][0] == 0, 'not starting at x=0'
            assert mesh.cells[
                'center'][-1] >= np.amax(mesh.cells['center'][-1])
            assert isinstance(
                mesh.group, mesh1d_group), 'can only belong to trivial group'
            mesh.group = self.group
            mesh.layer_i = i
            L = mesh.cells['center'][-1]
            mesh.cells['center'] += x
            mesh.internal.cells['center'] += x
            mesh.boundary.cells['center'] += x
            mesh.faces['center'] += x
            x += L
            self.domains[name] = mesh
