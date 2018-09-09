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

import matplotlib
import matplotlib.pylab as plt
import matplotlib.ticker
from collections import defaultdict
from oedes.fvm import mesh1d
import numpy as np
import itertools
from matplotlib.collections import LineCollection
from matplotlib.ticker import EngFormatter
from oedes.utils import meta_key, Units

__all__ = ['matplotlib', 'plt', 'forcontext', 'subplots', 'Settings']


class mesh1d_mpl(object):
    def __init__(self, fig, ax):
        self.fig = fig
        self.ax = ax
        self.cycler = itertools.cycle(matplotlib.rcParams['axes.prop_cycle'])
        self.yunits = set()
        self.xunits = set()

    def plot(self, items, *args, **kwargs):
        if 'color' not in kwargs:
            kwargs['color'] = next(self.cycler)['color']
        lines = []
        for item in items:
            meta, value = item
            mesh = meta.mesh
            face = meta.face
            if not isinstance(mesh, mesh1d):
                raise ValueError('mesh type not supported')
            if face:
                X = mesh.faces['center']
            else:
                X = mesh.cells['center']
            lines.append(np.vstack((X, value)).transpose())
            self.yunits.add(meta.unit)
        self.xunits.add(Units.meter)
        self.ax.add_collection(LineCollection(lines, *args, **kwargs))
        self.ax.autoscale()

    def settings(self, settings):
        yunits = ' / '.join(r'$\mathrm{{{}}}$'.format(u)
                            for u in sorted(self.yunits))
        xunits = ' / '.join(r'$\mathrm{{{}}}$'.format(u)
                            for u in sorted(self.xunits))

        if 'ylabel' not in settings:
            self.ax.set_ylabel(yunits)
        if 'xlabel' not in settings and 'xunit' not in settings:
            if self.xunits == set([Units.meter]):
                self.ax.set_xlabel('Position')
                self.ax.xaxis.set_major_formatter(EngFormatter(unit=xunits))
            else:
                self.ax.set_xlabel(xunits)
        for name in ['xlabel', 'ylabel', 'xlim', 'ylim', 'xscale', 'yscale']:
            if name in settings:
                getattr(self.ax, 'set_' + name)(settings[name])

        prefixes = {'n': 1e-9, r'\mu': 1e-6, 'u': 1e-6, 'm': 1e-3, None: 1.}

        def aunit(axobj, unit):
            try:
                pv = prefixes[unit]
            except BaseException:
                pv = float(unit)
            axobj.set_major_formatter(matplotlib.ticker.FuncFormatter(
                lambda x, pos: '{0:g}'.format(x / pv)))
            axobj.set_minor_formatter(matplotlib.ticker.NullFormatter())

        if 'xunit' in settings:
            aunit(self.ax.xaxis, settings['xunit'])
        if 'yunit' in settings:
            aunit(self.ax.yaxis, settings['yunit'])


import re


class Settings:
    """
    Plot presets, following naming
    yquantity[_yunit-if-not-basic]_xquantity[_xunit_if_not_basic][_[log_or_lin_y][log_or_lin_x]]
    """

    c_x_nm_loglin = dict(
        yscale='log',
        ylabel=r'$\mathrm{m^{-3}}$',
        xunit='n',
        xlabel='nm')
    c_cm_x_nm_loglin = dict(
        yscale='log',
        yunit=1e6,
        ylabel=r'$\mathrm{cm^{-3}}$',
        xunit='n',
        xlabel='nm')
    c_x_nm = dict(
        ylabel=r'$\mathrm{m^{-3}}$',
        xunit='n',
        xlabel='nm')
    c_cm_x_nm = dict(
        yunit=1e6,
        ylabel=r'$\mathrm{cm^{-3}}$',
        xunit='n',
        xlabel='nm')
    potential_x_nm = dict(ylabel=r'V', xunit='n', xlabel='nm')
    E_x_nm = dict(ylabel=r'$\mathrm{V \ m^{-1}}$', xunit='n', xlabel='nm')
    Ef_x_nm = dict(ylabel=r'eV', xunit='n', xlabel='nm')

    j_V = dict(ylabel=r'$\mathrm{A \ m^{-2}}$', xlabel='V')
    j_V_loglin = dict(
        ylabel=r'$\mathrm{A \ m^{-2}}$',
        xlabel='V',
        yscale='log')
    j_V_loglog = dict(
        ylabel=r'$\mathrm{A \ m^{-2}}$',
        xlabel='V',
        yscale='log',
        xscale='log')

    j_mAcm_V = dict(yunit=10, ylabel=r'$\mathrm{mA \ cm^{-2}}$', xlabel='V')

    j_t = dict(ylabel=r'$\mathrm{A \ m^{-2}}$', xlabel='s')
    j_t_loglog = dict(
        ylabel=r'$\mathrm{A \ m^{-2}}$',
        xlabel='s',
        xscale='log',
        yscale='log')
    j_t_us = dict(ylabel=r'$\mathrm{A \ m^{-2}}$', xlabel=r'$\mu s$', xunit='u')
    j_mAcm_t_us = dict(
        yunit=10,
        ylabel=r'$\mathrm{mA \ cm^{-2}}$',
        xlabel=r'$\mu s$',
        xunit='u')


class context_plotting:
    def __init__(self, ctx, fig=None, ax=None):
        self.out = ctx.output()
        self.model = ctx.model
        self.ctx = ctx
        self.fig = fig
        if ax is not None:
            self.ax = ax
        else:
            self.fig, self.ax = plt.subplots()
        self._impl = None

        self.default_settings_species = Settings.c_x_nm_loglin
        self.default_settings_potential = Settings.potential_x_nm
        self.default_settings_E = Settings.E_x_nm
        self.default_settings_Ef = Settings.Ef_x_nm
        self.default_settings_selector = dict()

    def getimpl(self, mesh):
        if self._impl is None:
            if mesh is None or isinstance(mesh, mesh1d):
                c = mesh1d_mpl
            elif hasattr(mesh, 'mpl_class'):
                c = mesh.mpl_class
            else:
                raise ValueError(
                    'do not know mpl_class of %s' %
                    mesh.__class__)
            self._impl = c(self.fig, self.ax)
        return self._impl

    def select(self, expr):
        r = re.compile(expr)
        grouped = defaultdict(list)
        with_label = 'label' in r.groupindex
        for k in sorted(self.out.keys()):
            match = r.match(k)
            if match:
                ent = (self.out[meta_key][k], self.out[k])
                if not with_label:
                    yield k, [ent]
                else:
                    grouped[match.group('label')].append(ent)
        for k in sorted(grouped.keys()):
            yield k, grouped[k]

    def _plot(self, parts, settings=None, *args, **kwargs):
        if not parts:
            return
        impl = self.getimpl(parts[0][0].mesh)
        impl.plot(parts, *args, **kwargs)
        if settings is not None:
            impl.settings(settings)

    def apply_settings(self, settings):
        if self._impl is not None:
            self._impl.settings(settings)

    def plot(self, keys, *args, **kwargs):
        plotlist = [(self.out[meta_key][k], self.out[k]) for k in keys]
        return self._plot(plotlist, *args, **kwargs)

    def selector(self, expr, *args, **kwargs):
        if 'settings' not in kwargs:
            kwargs['settings'] = self.default_settings_selector
        want_label = 'label' not in kwargs
        for legendkey, parts in self.select(expr):
            if want_label:
                kwargs['label'] = legendkey
            self._plot(parts, *args, **kwargs)

    def allspecies(self, *args, **kwargs):
        if 'settings' not in kwargs:
            kwargs['settings'] = self.default_settings_species
        self.selector(r'(.*\.)?(?P<label>.*)\.c$', *args, **kwargs)

    def species(self, prefix, *args, **kwargs):
        if 'settings' not in kwargs:
            kwargs['settings'] = self.default_settings_species
        self.selector(r'(.*\.)?%s\.c$' % re.escape(prefix), *args, **kwargs)

    def potential(self, *args, **kwargs):
        if 'settings' not in kwargs:
            kwargs['settings'] = self.default_settings_potential
        self.selector(r'(.*\.)?(?P<label>potential)$', *args, **kwargs)

    def E(self, *args, **kwargs):
        if 'settings' not in kwargs:
            kwargs['settings'] = self.default_settings_E
        self.selector(r'(.*\.)?(?P<label>E)$', *args, **kwargs)

    def Ef(self, *args, **kwargs):
        if 'settings' not in kwargs:
            kwargs['settings'] = self.default_settings_Ef
        self.selector(r'(.*\.)?(?P<label>.*)\.Ef$', *args, **kwargs)

    def teval(self, *functions, **kwargs):
        settings = kwargs.pop('settings', None)
        results = self.ctx.teval(*functions)
        want_label = 'label' not in kwargs
        for i in range(1, len(results)):
            if want_label:
                kwargs['label'] = str(functions[i])
            self.ax.plot(results[0], results[i], **kwargs)
        self.ax.set_xlabel(str(functions[0]))
        if settings:
            self.getimpl(None).settings(settings)

    def apply(self, settings):
        self.impl.apply(settings)


def subplots(c, *args, **kwargs):
    fig, u = plt.subplots(*args, **kwargs)

    def wrap(x):
        if isinstance(x, tuple):
            return tuple(wrap(y) for y in x)
        elif isinstance(x, list):
            return list(wrap(y) for y in x)
        else:
            return context_plotting(c, fig, x)
    return wrap(u)


def forcontext(*args, **kwargs):
    return context_plotting(*args, **kwargs)
