Optical models
==============

Photon energy
-------------

.. math::

   E_p = h \nu

with :math:`h` Planck constant and :math:`\nu` is photon frequency.

Coherent transfer matrix method
-------------------------------

Transfer matrix method a convenient way of modeling thin film stacks. It is assumed that layers are
stacked along :math:`x` axis, with :math:`x_{i,i+1}` being interface between layer :math:`i` and
layer :math:`i+1`. Optical properties of each layer are specified by wavelength dependent complex
refraction coefficient :math:`\tilde{n}_i(\lambda)`.

Optical field inside layer :math:`i` at given point along :math:`x` axis is specified by column vector
:math:`\left[E_{i}^{+}(x),E_{i}^{-}(x)\right]^T`, with :math:`E_{i}^{+}(x)` being complex amplitude
of forward traveling wave, and :math:`E_{i}^{-}(x)` being complex amplitude of backward traveling
wave.

Snell law is determines angles of propagation in each layer

.. math::

   n_0 \sin \theta_0 = n_i \sin \theta_i

where index 0 refers to medium before first layer. :math:`\theta_0` is angle of illuminating wave.
All angles can be complex numbers. Since :math:`\arcsin` is multivalued function, angle of forward
traveling wave is found from conditions that forward wave has forward pointing Poynting vector,
or alternatively, that the amplitude of forward wave decays in absorbing medium.

In this convention, interface between layers is described by matrix :math:`\mathbf{M}` as

.. math::

   \left[\begin{array}{c}
   E_{i}^{+}(x)\\
   E_{i}^{-}(x)
   \end{array}\right]=\mathbf{M_{i,i+1}}\left[\begin{array}{c}
   E_{i+1}^{+}(x)\\
   E_{i+1}^{-}(x)
   \end{array}\right]

with entries of matrix :math:`\mathbf{M}` specified as

.. math::

   \mathbf{M_{i,i+1}}=\frac{1}{t_{i,i+1}}\left[\begin{array}{cc}
   1 & r_{i,i+1}\\
   r_{i,i+1} & 1
   \end{array}\right]

where transmission coefficient :math:`t_{i,i+1}` and reflection coefficient :math:`r_{i,i+1}` are given by Fresnel equations for complex amplitudes of light passing from layer i to layer i+1. Coefficients for backward propagating wave :math:`t_{i+1,i}` and :math:`r_{i+1,i}` are eliminated using Stokes relations.

For s-polarized wave:

.. math::

   r_{i,i+1}=\frac{n_{i}\cos\theta_{i}-n_{i+1}\cos\theta_{i+1}}{n_{i}\cos\theta_{i}+n_{i+1}\cos\theta_{i+1}}

   t_{i,i+1}=\frac{2n_{i}\cos\theta_{i}}{n_{i}\cos\theta_{i}+n_{i+1}\cos\theta_{i+1}}

For p-polarized wave:

.. math::

   r_{i,i+1}=\frac{n_{i+1}\cos\theta_{i}-n_{i}\cos\theta_{i+1}}{n_{i+1}\cos\theta_{i}+n_{i}\cos\theta_{i+1}}

   t_{i,i+1}=\frac{2n_{i}\cos\theta_{i}}{n_{i+1}\cos\theta_{i}+n_{i}\cos\theta_{i+1}}

Propagation inside layer is described by matrix :math:`\mathbf{P}` as

.. math::

   \left[\begin{array}{c}
   E_{i}^{+}(x)\\
   E_{i}^{-}(x)
   \end{array}\right]=\mathbf{P_{i}(x)}\left[\begin{array}{c}
   E_{i}^{+}(x=x_{i,i+1})\\
   E_{i}^{-}(x=x_{i,i+1})
   \end{array}\right]

.. math::
   \mathbf{P_{i}(x)}=\left[\begin{array}{cc}
   \exp-i\delta_{i}(x) & 0\\
   0 & \exp i\delta_{i}(x)
   \end{array}\right]

with phase-shift

.. math::

   \delta_{i}(x)=\left(\frac{2\pi}{\lambda_{0}}\tilde{n}_{i}\cos\theta_{i}\right)\left(x_{i,i+1}-x\right)

Light entering layer :math:`i`, on side of layer :math:`i-1` has vector of complex amplitudes

.. math::

   \mathbf{v_k} \left( x=x_{k-1,k} \right) =\left(\Pi_{k\le i\le n}\mathbf{M_{i-1,i}}\mathbf{P_{i}}(x_{i-1,i})\right)\mathbf{M_{n,n+1}}\left[\begin{array}{c}
   t\\
   0
   \end{array}\right]

with vector :math:`\left[t,0\right]` denoting light leaving the device on the side opposite to illumination, with :math:`t` being complex amplitude of transmitted wave.

Applying above to whole device gives

.. math::

   \left[\begin{array}{c}
   1\\
   r
   \end{array}\right] = \mathbf{v_1}

with amplitude of illuminating wave set arbitrarily to :math:`1` and :math:`r` being complex amplitude of reflected wave.

When analyzing stack, firstly, solution :math:`t`, :math:`r` is found. Then intensity of light anywhere inside the device is calculated using found vectors :math:`\mathbf{v_i}` and propagation matrices :math:`\mathbf{P_i}`. Total intensity is found by applying Poynting formula. Absorbed energy is found by differentiating with respect to :math:`x`.

Incoherent light
----------------

Incoherent light is described by spectrum :math:`S(\lambda)`. Absorption of incoherent light is calculated as

.. math::

   A = \int A_{\mathrm{coherent}}\left(\lambda\right)S\left(\lambda\right)d\lambda

where :math:`A_{\mathrm{coherent}}\left(\lambda\right)` is calculated using coherent transfer matrix method.



