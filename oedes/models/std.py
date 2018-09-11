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
from .deprecated import BaseModel
from oedes.models.equations import BoltzmannDOS, BandTransport, AdvectionDiffusion, Electroneutrality, Poisson, NontransportedSpecies
from oedes.models.equations.general import species_v_D_charged_from_params
from oedes.models.equations import AppliedVoltage, FermiLevelEqualElectrode
from .source import LangevinRecombination, DirectRecombination, SRH, TrapSource, DirectGeneration
from oedes.fvm import mesh1d


def add_transport(model, mesh, z, name, traps=[],
                  dos_class=BoltzmannDOS, **kwargs):
    """
    Adds transport equation to the model.
    Also adds trap levels and source terms transport<->trap level.
    """

    eq = BandTransport(
        mesh=mesh,
        name=name,
        z=z,
        dos=dos_class(),
        thermal=model.thermal,
        **kwargs)
    eq.bc = [FermiLevelEqualElectrode(boundary)
             for boundary in mesh.boundaries]
    model.species.append(eq)
    basename = name
    for name in traps:
        teq = NontransportedSpecies(mesh, basename + '.' + name, z=z)
        model.species.append(teq)
        model.sources.append(TrapSource(eq, teq))
    return eq


def electronic_device(model, mesh, polarity, ntraps=[], ptraps=[], **kwargs):
    """
    Creates simple electronic device:
    - if Poisson's equation is not present, create default with voltage
    - creates 'hole' if 'p' polarity requested, optionally with corresponding traps
    - creates 'electron' if 'n' polarity requested, optionally with corresponding traps
    - if both polarities are present, adds recombination term
    """
    if model.poisson.mesh is None:
        model.poisson = Poisson(mesh)
        model.poisson.bc = [AppliedVoltage(boundary)
                            for boundary in mesh.boundaries]
    assert polarity in ['n', 'p', 'np', 'pn', ''], "invalid polarity"
    electron, hole = None, None
    if 'n' in polarity:
        electron = add_transport(
            model, mesh, -1, 'electron', traps=ntraps, **kwargs)
    if 'p' in polarity:
        hole = add_transport(model, mesh, +1, 'hole', traps=ptraps, **kwargs)
    if electron is not None and hole is not None:
        r = LangevinRecombination(electron, hole)
        model.sources.append(r)


def _find(model):
    return model.findeq('cation'), model.findeq('anion')


def _initial_salt(model, cinit, nc=1., na=1.):
    "Return initial state of device corresponding to uniformly distributed ions with salt concentration cinit"

    cation, anion = _find(model)
    assert nc > 0 and na > 0 and cation.z > 0 and anion.z < 0
    assert cation.z * nc == -anion.z * na
    x = np.zeros_like(model.X)
    x[cation.idx] = cinit * nc
    x[anion.idx] = cinit * na
    return x


def add_ions(model, mesh, zc=1, za=-1):
    assert zc > 0 and za < 0, "follow standard convention"
    if model.poisson.mesh is None:
        model.poisson = Poisson(mesh)
        model.poisson.bc = [AppliedVoltage(k) for k in mesh.boundaries]
    cation = AdvectionDiffusion(
        mesh=mesh,
        name='cation',
        z=zc,
        thermal=model.thermal,
        v_D=species_v_D_charged_from_params)
    anion = AdvectionDiffusion(
        mesh=mesh,
        name='anion',
        z=za,
        thermal=model.thermal,
        v_D=species_v_D_charged_from_params)
    model.species.extend([cation, anion])
    return cation, anion, lambda *args, **kwargs: _initial_salt(
        model, *args, **kwargs)


electrolyte = add_ions


def holeonly(L, traps=[]):
    model = BaseModel()
    electronic_device(model, mesh1d(L), 'p', ptraps=traps)
    model.setUp()
    return model


def electrononly(L, traps=[]):
    model = BaseModel()
    electronic_device(model, mesh1d(L), 'n', ntraps=traps)
    model.setUp()
    return model


def bulk_heterojunction(model, mesh, ptraps=[], ntraps=[], selective_contacts=False, absorption=lambda x: 1.,
                        langevin_recombination=False, const_recombination=False, srh_recombination=False):
    assert any(
        (langevin_recombination,
         const_recombination,
         srh_recombination))
    if model.poisson.mesh is None:
        model.poisson = Poisson(mesh)
        model.poisson.bc = [AppliedVoltage(k) for k in mesh.boundaries]
    electron = add_transport(model, mesh, -1, 'electron', traps=ntraps)
    hole = add_transport(model, mesh, 1, 'hole', traps=ptraps)
    if selective_contacts:
        electron.bc = [FermiLevelEqualElectrode('electrode1')]
        hole.bc = [FermiLevelEqualElectrode('electrode0')]
    model.sources.append(DirectGeneration([electron, hole], absorption))
    semiconductor = Electroneutrality([electron, hole], name='semiconductor')
    model.other = [semiconductor]
    if langevin_recombination:
        model.sources.append(LangevinRecombination(semiconductor))
    if const_recombination:
        model.sources.append(DirectRecombination(semiconductor))
    if srh_recombination:
        model.sources.append(SRH(semiconductor, name='srh'))
    return model


def bulk_heterojunction_params(
        barrier=0.3, bandgap=1.2, Nc=1e27, Nv=1e27, T=300., epsilon_r=5., mu_e=1e-10, mu_h=1e-10):
    p = {
        'T': T,
        'epsilon_r': epsilon_r,
        'electron.mu': mu_e,
        'hole.mu': mu_h,
        'electron.N0': Nc,
        'hole.N0': Nv,
        'generation.I': 1.,
        'electron.energy': 0.,
        'electrode0.workfunction': bandgap - barrier,
        'hole.energy': -bandgap,
        'electrode1.workfunction': barrier,
        'electrode0.voltage': 0.,
        'electrode1.voltage': 0.
    }
    return p
