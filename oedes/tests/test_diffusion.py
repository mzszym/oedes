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

# This tests if diffusion equation is solved correctly with different
# boundary conditions

import matplotlib.pylab as plt
from oedes.testing import store

make_plots = False

from oedes import solve, bdf1adapt, models, transientsolve
from oedes.fvm import mesh1d
from oedes.models import AppliedVoltage, Poisson, AdvectionDiffusion
from oedes import context
from oedes import testing
import numpy as np
import unittest
from oedes.models import Zero, Equal
from oedes import mpl


def integrate_c_internal(eq, c):
    return np.sum((eq.mesh.cells['volume'] * c)[eq.mesh.internal.idx])


def integrate_c_all(eq, c):
    return np.sum((eq.mesh.cells['volume'] * c))


def run_diffusion(L, times, bc_equation=None,
                  integrate_c=integrate_c_internal, rtol=1e-3, atol=0., return_context=False, **kwargs):
    b = models.BaseModel()
    mesh = mesh1d(L, dx_boundary=L * 1e-5)
    # Poisson's equation should be there
    b.poisson = Poisson(mesh)
    b.poisson.bc = [AppliedVoltage(boundary) for boundary in mesh.boundaries]
    # All species are uncharged

    def v_D(ctx, eq):
        return 0., kwargs[eq.name][0]
    s = dict()
    for k in sorted(kwargs.keys()):
        species = AdvectionDiffusion(mesh, k, z=0, v_D=v_D)
        b.species.append(species)
        if bc_equation is not None:
            species.bc = bc_equation(species)
        s[k] = species
    b.setUp()
    params = {'T': 300., 'electrode0.voltage': 0, 'electrode1.voltage': 0, 'electrode0.workfunction': 0,
              'electrode1.workfunction': 0, 'epsilon_r': 3.}
    xinit = b.X.copy()
    for k, v in kwargs.items():
        xinit[s[k].idx] = kwargs[k][1](s[k].mesh.cells['center'] / L)
    conservation = []

    def update_conservation(t, out):
        for eq in b.species:
            store(out[eq.name + '.c'], rtol=1e-5, atol=1e-20)
        cons = [integrate_c(eq, out[eq.name + '.c']) for eq in b.species]
        for species in b.species:
            conservation.append((t,) + tuple(cons))
    update_conservation(0., b.output(0, xinit, 0. * xinit, params))
    c = context(b, x=xinit)
    for ts in c.transientsolve(params, times):
        pass
    for ts in c.timesteps():
        update_conservation(ts.time, ts.output())
        if make_plots:
            ylim = [0., np.amax(ts.x) * 1.1]
            p = mpl.forcontext(ts)
            p.allspecies(settings={})
            plt.ylim(ylim)
            plt.title('t=%e' % ts.time)
            plt.show()
    conservation = np.asarray(conservation)
    for i, species in enumerate(b.species):
        assert np.allclose(
            conservation[:, i + 1], conservation[0, i + 1], rtol=rtol, atol=atol)
    if make_plots:
        for i, species in enumerate(b.species):
            plt.plot(conservation[:, 0], conservation[:, i + 1], 'o-')
        plt.xscale('log')
        plt.title('Conservation of species')
        plt.xlabel('Time')
        plt.ylabel('Total amount of species')
        plt.show()
    if return_context:
        return c, conservation


test_species = dict(part1=(1., lambda u: np.abs(u - 0.5) < 0.1), part2=(10., lambda u: np.abs(u - 0.2) < 0.1),
                    part3=(1e2, lambda u: np.where(u > 0.999, 1e2, 0)))


class TestDiffusion(unittest.TestCase):
    _multiprocess_can_split_ = True

    def test_isolated(self):
        return run_diffusion(1., bc_equation=lambda eq: [], integrate_c=integrate_c_all, times=[
                             1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 1e1], **test_species)

    def test_zero(self):
        return run_diffusion(1., bc_equation=lambda eq: [Zero('electrode0')], integrate_c=integrate_c_all, times=[
                             1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 1e1], atol=1e100, **test_species)

    def test_periodic(self):
        return run_diffusion(1., bc_equation=lambda eq: [Equal(eq, 'electrode1')], integrate_c=integrate_c_all, times=[
                             1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 1e1], **test_species)

    def test_periodic2(self):
        return run_diffusion(1., bc_equation=lambda eq: [Equal(eq, 'electrode0')], integrate_c=integrate_c_all, times=[
                             1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 1e1], part1=(1., lambda x: np.exp(-4 * (x - 0.25)**2)), part2=(5., lambda x: np.cos(20 * x)**4))

    def test_periodic3(self):
        return run_diffusion(1., bc_equation=lambda eq: [Equal(eq, 'electrode1')], integrate_c=integrate_c_all, times=[
                             1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 1e1], part1=(1., lambda x: np.exp(-4 * (x - 0.25)**2)), part2=(5., lambda x: np.cos(20 * x)**4))

    def test_mixed(self):
        return run_diffusion(1., bc_equation=lambda eq: [Zero('electrode1'), Equal(eq, 'electrode0')], integrate_c=integrate_c_all, times=[
                             1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 1e1], part1=(1., lambda x: np.exp(-4 * (x - 0.25)**2)), part2=(5., lambda x: np.cos(20 * x)**4), atol=1e100)

    def runTest(self):
        pass
