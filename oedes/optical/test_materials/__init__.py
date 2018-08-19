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

from oedes.optical.databases.refractiveindex import RefractiveIndexInfoMaterial, RefractiveIndexInfoMaterialWarning
import os
path = os.path.dirname(__file__)

Glass = RefractiveIndexInfoMaterial(
    os.path.join(
        path,
        'glass.yml'),
    ignore_warnings=True, name='glass')
Al = RefractiveIndexInfoMaterial(
    os.path.join(
        path,
        'al.yml'),
    ignore_warnings=True, name='aluminium')
PEDOT_PSS = RefractiveIndexInfoMaterial(
    os.path.join(path, 'pedot_pss.yml'), ignore_warnings=True, name='PEDOT:PSS')
P3HT_PC61BM = RefractiveIndexInfoMaterial(
    os.path.join(path, 'p3ht_pc61bm.yml'), ignore_warnings=True, name='P3HT:PCBM')
ITO = RefractiveIndexInfoMaterial(
    os.path.join(
        path,
        'ito.yml'),
    ignore_warnings=True, name='ITO')
