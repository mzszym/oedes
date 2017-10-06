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
from .dos import *
from .boundary import *
from .source import *
from oedes.fvm import mesh1d


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
        r = LangevinRecombination(
            mesh, electron_prefix=electron.prefix, hole_prefix=hole.prefix)
        model.sources.append(r)


def add_ions(model, mesh, zc=1, za=-1):
    assert zc > 0 and za < 0, "follow standard convention"
    cation = TransportCharged(mesh, 'cation', zc)
    anion = TransportCharged(mesh, 'anion', za)
    model.species.extend([cation, anion])
    for prefix in ['cation', 'anion']:
        model.species_v_D[prefix] = species_v_D_charged_from_params
    return cation, anion


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
