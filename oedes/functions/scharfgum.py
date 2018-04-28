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

from oedes import ad


def B(x):
    """
    Bernoulli function B(x)=x/(exp(x)-1), implemented for double precision
    calculation
    """

    # For small x, Taylor expansion should be used instead of direct
    # formula which will is not even defined for x=0.
    BP = 0.0031769106669272879

    # Taylor series
    x2 = x**2.
    taylor = 1. - x / 2. + x2 / 12. - x2**2 / 720.

    # Direct calculation
    # 1e-20 is added to avoid NaN for small x. It only modifies
    # the result if abs(x)<BP. In such a case, the direct formula
    # is not used anyway.
    # In order to avoid overflow errors, maximum x is limited to 700
    # to ensure that exp(xn) is always representable in double precision.
    # Anyway xn/(exp(xn)-1.) is almost 0. for large x
    xn = ad.where(x < 700., x, 700.)
    direct = xn / ((ad.exp(xn) - 1.) + 1e-20)

    # Breakpoint BP is chosen so that Taylor series and direct
    # calculation coincide in double precision.
    return ad.where(abs(x) < BP, taylor, direct)


def Aux1(x):
    """Auxiliary function Aux1(x) = x/sinh(x)
    Properties:
    Aux1(x)=Aux1(-x)
    Aux1(x/2)-x*Aux2(x/2)=B(x)
    """
    return Aux2(-x) * B(x) * 2.


def Aux2(x):
    """
    Auxiliary function Aux2(x) = 1./(exp(x)+1)
    Properties: Aux2(x)+Aux2(-x)=1
    """
    xn = ad.where(x < 700., x, 700.)
    return 1. / (ad.exp(xn) + 1.)


def ScharfetterGummelFlux(dr, cfrom, cto, v, D, full_output=False):
    """
    Calculate Scharfetter-Gummel approximation of flux between points
    separated by distance dr, with species concentrations cfrom, cto,
    drift velocity v and diffusion coefficient D.
    Returns: flux or (flux,c,dc/dx) in center, if full_output
    """
    u = D / dr
    t = v / u
    Bt = B(t)
    flux = (cfrom * B(-t) - cto * Bt) * u
    if not full_output:
        return flux
    t2 = 0.5 * t
    aux2 = Aux2(t2)
    aux1 = Bt + t * aux2
    c = cfrom * (1. - aux2) + cto * aux2
    dc = aux1 * (cto - cfrom) / dr
    return (flux, c, dc)
