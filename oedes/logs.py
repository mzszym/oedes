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

import logging
import sys
import time

oedes = logging.getLogger('oedes')
solver = logging.getLogger('oedes.solver')
timestepping = logging.getLogger('oedes.solver.timestepping')
nonlinear = logging.getLogger('oedes.solver.nonlinear')
nonlinear_convergence = logging.getLogger('oedes.solver.nonlinear.convergence')
linear = logging.getLogger('oedes.solver.linear')
models = logging.getLogger('oedes.models')
fvm = logging.getLogger('oedes.fvm')


class IndentedFormatter(logging.Formatter):
    def _indentLevel(self, name):
        if name.startswith('oedes.solver.timestepping'):
            return 0
        if name.startswith('oedes.solver.nonlinear'):
            if name.startswith('oedes.solver.nonlinear.convergence'):
                return 2
            if name.startswith('oedes.solver.nonlinear.timing'):
                return 2
            return 1
        if name.startswith('oedes.solver.linear'):
            return 2
        if name.startswith('oedes.models'):
            return 2
        if name.startswith('oedes.fvm'):
            return 2
        return 0

    def format(self, record):
        return '\t' * self._indentLevel(record.name) + record.msg


class Timer(object):
    def __init__(self):
        self.t0 = time.clock()
        self.t1 = None

    def elapsed(self):
        if self.t1 is None:
            self.stop()
        return self.t1 - self.t0

    def start(self):
        self.t0 = time.clock()

    def stop(self):
        self.t1 = time.clock()

    def __str__(self):
        return '%.3e s' % self.elapsed()


class LogOutput(object):
    def __init__(self, level, handler=None, logger=None):
        if handler is None:
            handler = logging.StreamHandler()
            handler.setLevel(level)
        if logger is None:
            logger = oedes
        self.handler = handler
        self.logger = logger
        self.level = level
        self.formatter = IndentedFormatter()

    def __enter__(self):
        self.handler.setFormatter(self.formatter)
        self.logger.addHandler(self.handler)

    def __exit__(self, *args):
        self.logger.removeHandler(self.handler)
