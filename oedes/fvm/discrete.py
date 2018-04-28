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
from oedes import logs


class ConvergenceTest(object):
    pass


class ElementwiseConvergenceTest(ConvergenceTest):

    def __init__(self, rtol=1e-6, atol=0., eps=0.):
        self.rtol = rtol
        self.atol = atol
        self.eps = eps

    def _testix(self, eq, ix, F, x, dx, report, prefix):
        i = eq.idx[ix]
        if len(i) == 0:
            return True
        F = F[i]
        x = x[i]
        dx = dx[i]
        converged_abs = np.abs(F) < self.atol
        converged_rel = np.abs(dx) < (self.rtol * np.abs(x) + self.eps)
        converged = np.logical_or(converged_abs, converged_rel).all()

        def amax(v):
            if len(v) == 0:
                return 0.
            else:
                return np.amax(v)
        with np.errstate(divide='ignore', invalid='ignore'):
            infodict = dict(converged=converged,
                            atol=self.atol,
                            rtol=self.rtol,
                            maxabs=amax(np.abs(F)),
                            maxabs_nc=amax(
                                np.abs(F[np.logical_not(converged_rel)])),
                            maxrel=amax(np.abs(dx / x)),
                            maxrel_nc=amax(np.abs((dx / x)[np.logical_not(converged_abs)])))
            report[prefix] = infodict
        logger = logs.nonlinear_convergence
        if logger.isEnabledFor(logs.logging.DEBUG):
            logger.debug('%s on %s:' % (self.__class__.__name__, prefix))

            def fmt(item):
                k, v = item
                if isinstance(v, float) or isinstance(v, np.ndarray):
                    sv = '%.3e' % v
                else:
                    sv = str(v)
                return '%s=%s' % (k, sv)
            logger.debug(' %s' % ' '.join(list(map(fmt, infodict.items()))))
        return converged

    def testEquation(self, eq, F, x, dx, report):
        ix_all = np.hstack(
            [eq.mesh.internal.idx, eq.mesh.boundary.idx[eq.bc_dof_is_free]])
        return self._testix(eq, ix_all, F, x, dx, report, eq.prefix)

    def testBoundary(self, eq, bc, F, x, dx, report):
        assert bc.owner_eq is eq
        return self._testix(eq, eq.mesh.boundary.idx[
                            bc.boundary['bidx']], F, x, dx, report, '.'.join([str(eq.prefix), bc.name]))


class DummyConvergenceTest(ConvergenceTest):
    def testEquation(self, eq, F, x, dx, report):
        return True

    def testBoundary(self, eq, bc, F, x, dx, report):
        return True


class DiscreteEquation(object):
    def __init__(self, name=None):
        self.convergenceTest = None
        self.idx = None
        self.transientvar = 1.
        self.name = name

    def init(self, builder):
        pass

    def update(self, x, *args):
        pass

    def scaling(self, xscaling, fscaling):
        pass

    def testConverged(self, F, x, dx, report):
        return self.convergenceTest.testEquation(self, F, x, dx, report)


class GeneralDiscreteEquation(DiscreteEquation):
    def __init__(self, ndof, name=None):
        super(GeneralDiscreteEquation, self).__init__(name=name)
        self._ndof = ndof

    def ndof(self):
        return self._ndof


class DelegateConvergenceTest(object):
    def __init__(self, target):
        self.target = target

    def testEquation(self, eq, F, x, dx, report):
        return self.target.convergenceTest.testEquation(
            eq, F, x, dx, report)

    def testBoundary(self, eq, bc, F, x, dx, report):
        return self.target.convergenceTest.testBoundary(
            eq, bc, F, x, dx, report)
