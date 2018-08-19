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
import scipy as sp
from oedes.functions import wavelength_to_photon_energy
from oedes.optical.func import wavelength_range
import scipy.interpolate

__all__ = ['IsotropicLayer', 'NormalIncidence']


def forward(n, angle):
    "Decide if angle is forward traveling plane wave in medium with index n"
    if not np.all(np.real(n) * np.imag(n) >= 0):
        raise ValueError('amplification of light not supported')

    # Decision on basis of direction of Poynting vector or lack of gain
    x = n * np.cos(angle)
    return np.where(np.abs(x.imag) > 0, x.imag > 0, x.real > 0)


def snell_forward(n, angle_of_incidence):
    """
    return list of angle theta in each layer based on angle th_0 in layer 0,
    using Snell's law
    """
    angles = sp.arcsin(n[0]*np.sin(angle_of_incidence) / n)
    return np.where(forward(n, angles), angles, np.pi-angles)


def _aux_calculations(polarization_is_s, n_i, n_f, th_i, th_f):
    c_i = np.cos(th_i)
    c_f = np.cos(th_f)
    k_i = n_i * c_i
    k_f = n_f * c_f
    return dict(c_i=c_i, c_f=c_f, k_i=k_i, k_f=k_f)


def _power_calculation(r, t, polarization_is_s, n_i, n_f, aux):
    k_i_c = n_i * np.conj(aux['c_i'])
    k_f_c = n_f * np.conj(aux['c_f'])
    aux.update(k_i_c=k_i_c, k_f_c=k_f_c)
    r_ = np.abs(r)
    R = r_*r_
    t_ = np.abs(t)
    T = t_*t_ * np.where(polarization_is_s,
                         aux['k_f'].real / aux['k_i'].real,
                         k_f_c.real / k_i_c.real)
    return R, T


class FresnelCalculation(object):
    def __init__(self, polarization_is_s, n_i, n_f,
                 th_i, th_f, with_power=False):
        def _do(c_i, c_f, k_i, k_f):
            u_i = n_i * c_f
            u_f = n_f * c_i
            self.r = np.where(polarization_is_s,
                              (k_i - k_f) / (k_i + k_f),
                              (u_f - u_i) / (u_f + u_i))
            self.t = 2 * k_i / \
                np.where(polarization_is_s, k_i + k_f, u_f + u_i)
        self.aux = _aux_calculations(polarization_is_s, n_i, n_f, th_i, th_f)
        _do(**self.aux)
        if with_power:
            self.R, self.T = _power_calculation(
                self.r, self.t, polarization_is_s, n_i, n_f, self.aux)
            if with_power:
                self.R, self.T = _power_calculation(
                    self.r, self.t, polarization_is_s, n_i, n_f, self.aux)


class PowerCalculation(object):
    def __init__(self, r, t, polarization_is_s, n_i, n_f, th_i, th_f):
        aux = _aux_calculations(polarization_is_s, n_i, n_f, th_i, th_f)
        self.R, self.T = _power_calculation(
            r, t, polarization_is_s, n_i, n_f, aux)
        k_i, k_i_c = aux['k_i'], aux['k_i_c']
        self.power_entering = np.where(polarization_is_s,
                                       (k_i*(1+np.conj(r))*(1-r)).real / k_i.real,
                                       (k_i_c*(1.+r)*(1.-np.conj(r))).real / k_i_c.real)

class AbsorptionCalculation(object):
    def __init__(self, kz, z, vw, polarization_is_s, n, n0, theta, theta0, all_values=True):
        # Amplitude of forward-moving wave is Ef, backwards is Eb
        Ef = vw[0] * np.exp(1j * kz * z)
        Eb = vw[1] * np.exp(-1j * kz * z)
        c0 = np.cos(theta0)
        c = np.cos(theta)
        ds = 1. / (n0*c0).real
        dp = 1. / (n0*np.conj(c0)).real
        self.poyn = np.where(polarization_is_s, (n*c*np.conj(Ef+Eb)*(Ef-Eb)).real * ds,
                        (n*np.conj(c)*(Ef+Eb)*np.conj(Ef-Eb)).real * dp)
        if not all_values:
            return
        p_ = abs(Ef+Eb)
        q_ = abs(Ef-Eb)
        p = p_*p_
        q = q_*q_
        self.absor = np.where(polarization_is_s, (n*c*kz*p).imag *
                         ds, (n*np.conj(c)*(kz*q-np.conj(kz)*p)).imag * dp)
        Ex = np.where(polarization_is_s, 0, (Ef - Eb) * c)
        Ey = np.where(polarization_is_s, Ef + Eb, 0)
        Ez = np.where(polarization_is_s, 0, (-Ef - Eb) * np.sin(theta))
        self.E = (Ex, Ey, Ez)
        self.z = z


def mat2x2(a, b, c, d):
    return ((a, b), (c, d))


def dot(X, Y):
    (a1, b1), (c1, d1) = X
    (a2, b2), (c2, d2) = Y
    return mat2x2(a1*a2+b1*c2, a1*b2+b1*d2, c1*a2+d1*c2, c1*b2+d1*d2)


def dotvec(M, x):
    (a, b), (c, d) = M
    v, w = x
    return (a*v+b*w, c*v+d*w)


def transfer_matrix(thickness, n, polarization_is_s, theta_0, wavelength):
    if thickness[0] != np.inf or thickness[-1] != np.inf:
        raise ValueError('first and last layer should be infinite')
    thickness = thickness.copy()
    thickness[0] = -1.
    thickness[-1] = -1.

    kx = n[0]*np.sin(theta_0)
    if not np.all(kx.imag <= 100*np.abs(kx)*np.finfo(kx.dtype).eps):
        raise ValueError('n*theta[0] must be real in first layer')
    if not np.all(forward(n[0], theta_0)):
        raise ValueError('theta0 must be forward')
    theta = snell_forward(n, theta_0)
    kz = 2 * np.pi * n * np.cos(theta) / wavelength
    delta = kz * thickness[..., np.newaxis]
    num_layers = len(thickness)

    def fresnel_matrix(i):
        f = FresnelCalculation(polarization_is_s, n[i], n[i+1],
                               theta[i], theta[i+1])
        u = 1./f.t
        return mat2x2(u, u*f.r, u*f.r, u)

    def propagation_matrix(i):
        return mat2x2(np.exp(-1j*delta[i]), 0, 0, np.exp(1j*delta[i]))

    matrices = [fresnel_matrix(
        0)]+[dot(propagation_matrix(i), fresnel_matrix(i)) for i in range(1, num_layers-1)]
    Mfull = matrices[0]
    for M in matrices[1:]:
        Mfull = dot(Mfull, M)

    # Net complex transmission and reflection amplitudes
    (a, b), (c, d) = Mfull
    t = 1./a
    r = c*t

    amplitudes = [(t, np.zeros_like(t))]
    for i in range(num_layers-2, 0, -1):
        amplitudes.append(dotvec(matrices[i], amplitudes[-1]))
    amplitudes.append(np.zeros_like(np.asarray(amplitudes[0])))
    vw_list = np.asarray(list(map(np.asarray, amplitudes[::-1])))

    p = PowerCalculation(r, t, polarization_is_s,
                         n[0], n[-1], theta[0], theta[-1])

    return dict(r=r, t=t, M_list=matrices, th_list=theta, kz_list=kz, num_layers=num_layers,
                vw_list=vw_list, R=p.R, T=p.T, power_entering=p.power_entering)


class IsotropicLayer(object):
    def __init__(self, thickness, material):
        self.thickness = thickness
        self.material = material

    def refractive_index(self, wavelength):
        return self.material.refractive_index(wavelength)


class TransferMatrixResult:
    def __init__(self, layers, polarization_is_s, theta_0, wavelength):
        """
        Run transfer matrix calculation, with layers, and with (polarization_is_s, theta_incidence, wavelength) broadcast against each other
        Layer before first layer, and layer after last is vacuum/air with n=1
        """
        thickness = np.asarray(
            [np.inf] + [l.thickness for l in layers] + [np.inf])
        #polarization_is_s, wavelength, theta_0 = np.broadcast_arrays(polarization_is_s, wavelength, theta_0)
        n = np.asarray([np.ones_like(wavelength)] +
                       [l.refractive_index(wavelength) for l in layers] +
                       [np.ones_like(wavelength)])
        self.result = transfer_matrix(
            thickness, n, polarization_is_s, theta_0, wavelength)
        self.layer_dict = dict((id(layer), i+1)
                               for i, layer in enumerate(layers))
        self.n = n
        self.polarization_is_s = polarization_is_s
        a = AbsorptionCalculation(self.result['kz_list'], 0., (self.result['vw_list'][:,0,:],self.result['vw_list'][:,1,:]), polarization_is_s, n, n[0], self.result['th_list'], self.result['th_list'][0], all_values=False)
        self.incident_power_by_layer = a.poyn[1:]
        self.T = self.result['T']
        self.R = self.result['R']


    def absorption(self, layer, x):
        assert np.all(x >= 0)
        assert np.all(x <= layer.thickness)
        layer_number = self.layer_dict[id(layer)]
        th_list = self.result['th_list']
        return AbsorptionCalculation(self.result['kz_list'][layer_number], x[:, np.newaxis], self.result['vw_list'][layer_number],
                                     self.polarization_is_s, self.n[layer_number], self.n[0], th_list[layer_number], th_list[0])

    def layer_index(self, layer):
        return self.layer_dict[id(layer)]


class NormalIncidenceResultForLayer(object):
    def __init__(self, ni, layer, x):
        self.wavelengths = ni.wavelengths
        self.density = ni.density
        self.tmm = ni.tmm
        self.result = self.tmm.absorption(layer, x)
        self.layer = layer
        self._integrate = ni.integrate
        self.x = x

    def light_power(self):
        return self._integrate(self.result.poyn)

    def absorption(self):
        return self._integrate(self.result.absor)

    def photons_absorbed(self):
        return self._integrate(
            self.result.absor/wavelength_to_photon_energy(self.wavelengths))

    def interpolated(self, y):
        return scipy.interpolate.interp1d(
            self.x, y, kind='linear', bounds_error=True)

    absorption_in_photons = photons_absorbed


class NormalIncidence:
    def __init__(self, layers, spectrum, wavelengths, clip_wavelengths=True):
        if clip_wavelengths:
            a, b = wavelength_range(
                spectrum, *(layer.material.refractive_index for layer in layers))
            wavelengths = wavelengths[(wavelengths >= a) & (wavelengths <= b)]
        self.wavelengths = wavelengths
        self.density = spectrum(wavelengths)
        self.tmm = TransferMatrixResult(layers, True, 0, wavelengths)
        self.layers = layers
        self.incident_power_by_layer = self.integrate(self.tmm.incident_power_by_layer)
        self.T = self.integrate(self.tmm.T)
        self.R = self.integrate(self.tmm.R)

    def in_layer(self, layer, x):
        return NormalIncidenceResultForLayer(self, layer, x)

    def total_absorption_in_layer(self, layer):
        i = self.tmm.layer_index(layer)
        return self.incident_power_by_layer[i-1]-self.incident_power_by_layer[i]

    def integrate(self, y, axis=None):
        return np.trapz(y*self.density, self.wavelengths, axis=y.ndim-1)

    def illuminating_power(self):
        return self.integrate(np.asarray(1))
