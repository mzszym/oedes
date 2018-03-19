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

from .base_model import BaseModel
from .dos import *
from .boundary import *
from .source import *
from oedes.fvm import mesh1d
from oedes.ad import exp


def add_transport(model, mesh, z, prefix, traps=[], dos_class=BoltzmannDOS):
    """
    Adds transport equation to the model.
    Also adds trap levels and source terms transport<->trap level.
    """

    eq = TransportCharged(mesh, prefix, z=z)
    model.species.append(eq)
    if dos_class is not None:
        model.species_dos[eq.prefix] = dos_class()
    eq.bc = [FermiLevelEqualElectrode(boundary)
             for boundary in mesh.boundaries]
    model.species_v_D[eq.prefix] = species_v_D_charged_from_params
    for name in traps:
        teq = TransportCharged(mesh, eq.prefix + '.' + name, z=z)
        model.species.append(teq)
        model.species_v_D[teq.prefix] = species_v_D_not_transported
        #teq.bc = [Zero(boundary) for boundary in mesh.boundaries]
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
    if model.poisson is None:
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


def add_ions(model, mesh, zc=1, za=-1):
    assert zc > 0 and za < 0, "follow standard convention"
    if model.poisson is None:
        model.poisson = Poisson(mesh)
        model.poisson.bc = [AppliedVoltage(k) for k in mesh.boundaries]
    cation = TransportCharged(mesh, 'cation', zc)
    anion = TransportCharged(mesh, 'anion', za)
    model.species.extend([cation, anion])
    for prefix in ['cation', 'anion']:
        model.species_v_D[prefix] = species_v_D_charged_from_params
    return cation, anion


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
    if model.poisson is None:
        model.poisson = Poisson(mesh)
        model.poisson.bc = [AppliedVoltage(k) for k in mesh.boundaries]
    electron = add_transport(model, mesh, -1, 'electron', traps=ntraps)
    hole = add_transport(model, mesh, 1, 'hole', traps=ptraps)
    if selective_contacts:
        electron.bc = [FermiLevelEqualElectrode('electrode1')]
        hole.bc = [FermiLevelEqualElectrode('electrode0')]
    for p in ['electron', 'hole']:
        model.species_v_D[p] = species_v_D_charged_from_params
    model.sources.append(DirectGeneration([electron, hole], absorption))
    if langevin_recombination:
        model.sources.append(LangevinRecombination(electron, hole))
    if const_recombination:
        model.sources.append(DirectRecombination(electron, hole))
    if srh_recombination:
        model.sources.append(SRH(electron, hole, 'srh'))
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
        'absorption.I': 1.,
        'electron.level': 0.,
        'electrode0.workfunction': bandgap - barrier,
        'hole.level': bandgap,
        'electrode1.workfunction': barrier,
        'electrode0.voltage': 0.,
        'electrode1.voltage': 0.,
        'npi': Nc * Nv * exp(-bandgap / functions.ThermalVoltage(T))
    }
    return p
