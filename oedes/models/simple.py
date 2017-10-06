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


def evaluate_mu(eq, vars):
    "Calculate mu, allowing that it is provided as a function"
    mu = vars['params'][eq.prefix + '.mu']
    if callable(mu):
        mu = mu(T=vars['T'], c=vars['c'][eq.prefix], E=None)
    vars['mu'][eq.prefix] = mu
    return mu


def species_v_D_charged_from_params(eq, vars):
    """
    Default implementation of transported species:
    - assumes constant mobility, takes from parameters
    - assumes diffusion described by Einstein's relation
    Note that mobility is saved for other calculations (such as recombination).
    """
    mu = evaluate_mu(eq, vars)
    if isscalar(mu):
        facemu = mu
    else:
        facemu = eq.mesh.faceaverage(mu)
    v = eq.v(eq.mesh.faceaverage(mu), vars['E'])
    D = eq.mesh.faceaverage(eq.D(mu, vars['Vt']))
    return v, D


def species_v_D_FrenkelPoole(eq, vars):
    mu0 = vars['params'][eq.prefix + '.mu0']
    gamma = vars['params'][eq.prefix + '.gamma']
    v = gamma * sqrt(vars['Ecellm'] + 1e-10)
    v_max = np.log(1e10)
    v_min = -v_max
    v = where(v < v_max, v, v_max)
    v = where(v > v_min, v, v_min)
    mu = mu0 * exp(v)
    vars['mu'][eq.prefix] = mu
    v = eq.v(eq.mesh.faceaverage(mu), vars['E'])
    D = eq.mesh.faceaverage(eq.D(mu, vars['Vt']))
    return v, D


def species_v_D_not_transported(eq, vars):
    """
    Not transported species
    """
    return None


def simple_doping(vars):
    params = vars['params']
    return (params['Nd'] - params['Na']) * scipy.constants.elementary_charge


def _check_samemesh(equations):
    for eq in equations[1:]:
        assert eq.mesh is equations[
            0].mesh, 'different mesheshes not supported'


def _check_samevars(eqvars, equations):
    vars = eqvars[id(equations[0])]
    for eq in equations[1:]:
        assert eqvars[id(eq)] is vars, 'equations in different models'


def species_v_D_limited_upwind(eq, vars):
    v, D = species_v_D_charged_from_params(eq, vars)
    c = where(v > 0,
              vars['c'][eq.prefix][eq.mesh.faces['+']],
              vars['c'][eq.prefix][eq.mesh.faces['-']])
    v = v * (1. - c / vars['params'][eq.prefix + '.N0'])
    return v, D
