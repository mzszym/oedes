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

from oedes.functions import *
from oedes import ad
import numpy as np
from scipy.optimize import check_grad
from numpy.testing import assert_allclose


def test_scharfgum():
    x = np.array([-1e3, -1e2, -1e1, -1e0, -1e-1, -1e-2, -
                  1e-3, 0, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3])
    b_x = np.array([1.00000000e+03, 1.00000000e+02, 1.00004540e+01,
                    1.58197671e+00, 1.05083319e+00, 1.00500833e+00,
                    1.00050008e+00, 1.00000000e+00, 9.99500083e-01,
                    9.95008333e-01, 9.50833194e-01, 5.81976707e-01,
                    4.54019910e-04, 3.72007598e-42, 0.00000000e+00])
    aux1_x = np.array([0.00000000e+00, 7.44015195e-42, 9.07998597e-04,
                       8.50918128e-01, 9.98335276e-01, 9.99983334e-01,
                       9.99999833e-01, 1.00000000e+00, 9.99999833e-01,
                       9.99983334e-01, 9.98335276e-01, 8.50918128e-01,
                       9.07998597e-04, 7.44015195e-42, 0.00000000e+00])
    aux2_x = np.array([1.00000000e+00, 1.00000000e+00, 9.99954602e-01,
                       7.31058579e-01, 5.24979187e-01, 5.02499979e-01,
                       5.00250000e-01, 5.00000000e-01, 4.99750000e-01,
                       4.97500021e-01, 4.75020813e-01, 2.68941421e-01,
                       4.53978687e-05, 3.72007598e-44, 0.00000000e+00])
    assert_allclose(B(x), b_x, rtol=1e-6, atol=1e-300)
    assert_allclose(Aux1(x), aux1_x, rtol=1e-6, atol=1e-300)
    assert_allclose(Aux2(x), aux2_x, rtol=1e-6, atol=1e-300)
    assert_allclose(ScharfetterGummelFlux(
        1., 2., 1., 100., 1., False), 200.)
    flux, midc, dc = ScharfetterGummelFlux(1., 2., 1., 100., 1., True)
    assert_allclose(flux, 200.)
    assert_allclose(midc, 2.)
    assert_allclose(dc, -1.92874985e-20, atol=0.)


def check_gdos_impl(impl):
    I = np.exp(np.linspace(np.log(impl.I_min), np.log(impl.I_max), 20))
    assert impl.a_max >= np.sqrt(2.) * 6
    a = impl.a_max
    b = impl.b(a, I)
    I1 = impl.I(a, b)
    assert_allclose(I, I1, atol=0., rtol=1e-12)
    g = gdos.diffusion_enhancement(impl.a_max, I, impl)
    assert_allclose(g[0], 1., atol=0., rtol=1e-5)
    c = np.array([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 5e-1])
    g6 = np.array([1.15279545, 1.22027379, 1.31057169, 1.43379655, 1.6093657,
                   1.879881, 2.36265441, 3.57370846, 7.84830035])
    assert_allclose(gdos.diffusion_enhancement(
        np.sqrt(2.) * 6., c, impl), g6, rtol=0.05, atol=0.)


def test_gdos():
    check_gdos_impl(gdos.defaultImpl)


def test_egdm():
    c = np.array([1e-8, 1e-6, 1e-4, 1e-2, 1e-1])
    g1_6 = np.array([2.06714517e+00, 4.92642645e+00, 3.31703853e+01,
                     2.18503730e+03, 8.88447976e+04])
    assert_allclose(egdm.g1(6., c), g1_6, rtol=1e-6, atol=0.)
    En = np.array([0., 0.5, 1., 1.5, 2.0])
    g2_6 = np.array([1., 1.69015385, 6.54410149, 40.54260014,
                     320.61218011])
    assert_allclose(egdm.g2(6., En), g2_6, rtol=1e-6, atol=0.)
    g3_6 = np.array(
        [1.15279545, 1.31057169, 1.6093657, 2.36265441, 3.57370846])
    assert_allclose(egdm.g3(6., c), g3_6, rtol=1e-2, atol=0.)
    Vt = physics.ThermalVoltage(300.)
    assert_allclose(egdm.nsigma_inverse(6., Vt), 0.1551119460699098, rtol=1e-6)
    assert_allclose(egdm.nsigma(0.1551119460699098, Vt),
                    6., rtol=1e-6, atol=0.)
