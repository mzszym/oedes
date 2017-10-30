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

from oedes import model
from oedes.ad import sparsesum
from oedes.fvm import Poisson, Transport
import numpy as np
import scipy.sparse
import itertools
import warnings
from oedes.util import TODOWarning
import logging
import scipy.sparse.csgraph


class PrefixDict:
    def __init__(self, d, prefix):
        if isinstance(d, PrefixDict):
            self.d = d.d
            self.prefix = d.prefix + '.' + prefix
        else:
            self.d = d
            self.prefix = prefix

    def _addprefix(self, k):
        if self.prefix is None:
            return k
        if k.startswith('__'):
            return k
        return self.prefix + '.' + k

    def __getitem__(self, k):
        k = self._addprefix(k)
        return self.d[k]

    def __setitem__(self, k, v):
        k = self._addprefix(k)
        return self.d.__setitem__(k, v)

    def update(self, d):
        for k in d:
            self[k] = d[k]

    def __contains__(self, k):
        return self.d.__contains__(self._addprefix(k))


def with_prefix(d, prefix):
    if d is None:
        return None
    return PrefixDict(d, prefix)


class vars_dict():

    def __init__(self, topvars, modelvars):
        self.topvars = topvars
        self.modelvars = modelvars

    def __getitem__(self, k):
        if k in self.modelvars:
            return self.modelvars[k]
        else:
            return self.topvars[k]

    def __setitem__(self, k, v):
        assert k not in self.modelvars
        self.modelvars[k] = v

    def update(self, d):
        for k in d:
            self[k] = d[k]

    def __contains__(self, k):
        return k in self.topvars or k in self.modelvars


logger = logging.getLogger('oedes.models.convergence')


class ComposableModel(model):
    name = None

    def __init__(self):
        model.__init__(self)
        self.after_load = []
        self.after_generate = []
        self.name = None

    @property
    def all_equations(self):
        "Return all equation in this model (and submodels)"
        raise NotImplementedError()

    def init_equations(self):
        for f in self.after_load:
            f.setUp(self)
        for f in self.after_generate:
            f.setUp(self)

    def load_equations(self, vars):
        raise NotImplementedError()

    def generate_residuals_equations(self, vars):
        raise NotImplementedError()

    def scaling(self, params):
        raise NotImplementedError('this is currently useless')

    def converged(self, *args):
        raise NotImplementedError('should not be called')

    def findeq(self, name):
        return dict((eq.prefix, eq) for eq in self.equations)[name]

    def _findeqs(self, test, prefix, return_names):
        for eq in self.equations:
            if test(eq):
                if return_names:
                    yield (prefix + eq.prefix, eq)
                else:
                    yield eq

    def findeqs(self, tests, return_names=False):
        return self._findeqs(tests, '', return_names=return_names)


class CompositeModel(ComposableModel):
    sub = None
    name = None
    after_load = None
    after_generate = None

    def __init__(self):
        self.sub = []
        self.name = None
        self.after_load = []
        self.after_generate = []

    @property
    def all_equations(self):
        return itertools.chain(
            self.equations, *(s.all_equations for s in self.sub))

    def create_indexes(self):
        i = 0
        for eq in self.all_equations:
            assert eq.idx is None, 'equation already has indices in unknown vector'
            eq.idx = np.arange(i, i + eq.mesh.ncells)
            i += eq.mesh.ncells
        self.X = np.zeros(i)

    def create_boundaries(self):
        i = [np.zeros(0, dtype=np.int)]
        j = [np.zeros(0, dtype=np.int)]
        for eq in self.all_equations:
            fdof = np.ones(len(eq.mesh.boundary.idx), dtype=np.bool)
            for bc in eq.bc:
                bc.setUp(eq)
                fdof[bc.bidx] = False
                valid = bc.conservation_to >= 0
                i.append(eq.idx[eq.mesh.boundary.idx[bc.bidx]][valid])
                j.append(bc.conservation_to[valid])
            eq.bc_free_dof = np.arange(len(eq.mesh.boundary.idx))[fdof]
        i = np.hstack(i)
        j = np.hstack(j)
        # optimize: should only create vector for boundary items
        g = scipy.sparse.csr_matrix(
            (np.ones_like(i), (i, j)), shape=(len(self.X),) * 2)
        nlabels, labels = scipy.sparse.csgraph.connected_components(
            g, directed=False, return_labels=True)
        self.bc_label_volume = sparsesum(nlabels, ((labels[eq.idx], eq.mesh.cells[
                                         'volume']) for eq in self.all_equations))
        self.bc_labels = labels
        for eq in self.all_equations:
            eq.boundary_labels = self.bc_labels[eq.idx[eq.mesh.boundary.idx]]

    def setUp(self):
        assert self.X is None, 'already set-up'
        self.create_indexes()
        for s in self.sub:
            s.init_equations()
        self.init_equations()
        self.create_boundaries()
        self.transientvar = self.X.copy()
        for eq in self.all_equations:
            self.transientvar[eq.idx] = eq.transientvar

    def load(self, time, x, xt, params, full_output, toplevel, solver):
        subvars = dict()
        modelvars = dict(time=time, x=x, xt=xt, solver=solver, model=self,
                         params=params, full_output=full_output, subvars=subvars)
        vars = vars_dict(toplevel, modelvars)
        # Set eqvars
        toplevel['modelvars'][id(self)] = vars
        eqvars = toplevel['eqvars']
        for eq in self.all_equations:
            eqvars[id(eq)] = modelvars
        # Load submodels
        for s in self.sub:
            assert s.name not in subvars, 'duplicate submodel name %s' % s.name
            subvars[s.name] = s.load(time, x, xt, params=with_prefix(
                params, s.name), full_output=with_prefix(full_output, s.name), toplevel=toplevel, solver=solver)
        # Load and evaluable variables equations in this model
        self.load_equations(vars)
        # Do additional calculations, which depend on
        for f in self.after_load:
            f.run(self, vars)
        return vars

    def generate_residuals(self, vars):
        # Return residuals, firslty submodels
        for s in self.sub:
            for res in s.generate_residuals(vars['subvars'][s.name]):
                yield res
        for res in self.generate_residuals_equations(vars):
            yield res
        for f in self.after_generate:
            f.run(self, vars)
        # return
        # itertools.chain(itertools.chain(s.generate_residuals(vars['subvars'][s.name])
        # for s in self.sub),self.generate_residuals(vars))

    def evaluate(self, time, x, xt, params, full_output=None, solver=None):
        vars = self.load(time, x, xt, params, full_output=full_output, toplevel=dict(
            bc_conservation=[], eqvars=dict(), modelvars=dict()), solver=solver)
        return itertools.chain(self.generate_residuals(
            vars), self.generate_bc_conservation(vars))

    def generate_bc_conservation(self, loc):
        bc_label_conservation = sparsesum(
            len(self.bc_labels), loc['bc_conservation'])
        for eq in self.all_equations:
            i = eq.idx[eq.mesh.boundary.idx[eq.bc_free_dof]]
            j = self.bc_labels[i]
            yield i, bc_label_conservation[j] * (1. / self.bc_label_volume[j])

    @property
    def equations(self):
        return []

    def generate_residuals_equations(self, vars):
        return []

    def load_equations(self, vars):
        pass

    def update_equations(self, x, params):
        pass

    def converged_equations(self, F, x, dx, params, report):
        return all([eq.testConverged(F, x, dx, params, report)
                    for eq in self.equations])

    # Things to be corrected!
    def scaling(self, params):
        # warnings.warn(
        #    'scaling  : should be corrected, nested, does not distinguish types of equations (BCs vs conservation)', TODOWarning)
        xscaling = np.ones_like(self.X)
        fscaling = np.ones_like(self.X)
        lunit = 1e-9
        tunit = 1e-12
        for eq in self.all_equations:
            if isinstance(eq, Poisson):
                fscaling[eq.idx] = lunit
            elif isinstance(eq, Transport):
                xscaling[eq.idx] = lunit**-3
                fscaling[eq.idx] = lunit * tunit
        return xscaling, fscaling

    def update(self, x, params):
        self.update_equations(x, params)
        for s in self.sub:
            sparams = with_prefix(params, s.name)
            s.update(x, sparams)

    def _converged(self, F, x, dx, params, report):
        return all([self.converged_equations(F, x, dx, params, report)] +
                   [sub._converged(F, x, dx, PrefixDict(params, sub.name), PrefixDict(report, sub.name)) for sub in self.sub])

    def converged(self, residuals, x, dx, params, report):
        c = self._converged(residuals, x, dx, params, report)
        for k in sorted(report.keys()):
            v = report[k]
            logger.debug('-------- %s converged=%r --------' %
                         (k, v['converged']))
            logger.debug('atol=%r, max(|F|)=%e (noncon.: %e)' %
                         (v['atol'], v['maxabs'], v['maxabs_nc']))
            logger.debug('rtol=%r, max(|x/dx|)=%e (noncon.: %e)' %
                         (v['rtol'], v['maxrel'], v['maxrel_nc']))
        return c

    def findeq(self, name):
        parts = name.split('.', 1)
        if len(parts) > 1:
            return dict((s.name, s)
                        for s in self.sub)[parts[0]].findeq(parts[1])
        else:
            return ComposableModel.findeq(self, name)

    def _findeqs(self, tests, prefix, return_names):
        return itertools.chain(ComposableModel._findeqs(self, tests, prefix, return_names),
                               *(s._findeqs(tests, prefix + s.name + '.', return_names) for s in self.sub))
