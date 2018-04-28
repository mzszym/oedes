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

import oedes


def test_no_garbage():
    import gc
    gc.collect()
    gc.disable()
    gc.set_debug(gc.DEBUG_SAVEALL)

    def run():
        model, params = oedes.testing.unipolar(100e-9)
        model.setUp()
        c = oedes.context(model)
        c.solve(params)
        c.output()
    run()
    gc.collect()
    assert not gc.garbage
    gc.set_debug(0)
    gc.enable()
