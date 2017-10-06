Introduction
============

`oedes` simulates electrical processes in thin film electronic devices using the drift-diffusion model and optical properties using the transfer matrix approach.

The electrical model is defined as follows:

.. math::
   E=-\frac{\partial \phi}{\partial x}

   \frac{\partial}{\partial x}\left(\epsilon E\right)=q\sum z_{i}c_{i}

   j_{i}=\mu_{i}\left(c_{i}z_{i}E-\frac{D_{i}}{\mu_{i}}\frac{\partial c_{i}}{\partial x}\right)

   \frac{\partial c_i}{\partial  t} + \frac{\partial j_i}{\partial x} = s_i

where index :math:`i` refers to species index, :math:`z` is species charge in units of elementary charge, :math:`c` is species concentration and :math:`\mu` is mobility. :math:`s` denotes source term, which may contain subterms due to generation, recombination and trapping processes.

In bipolar devices, source term :math:`s` contains contribution of Langevin recombination, which is given by formula

.. math::

   R = \frac{e}{\epsilon} \left( c_n c_p - c_n^0 c_p^0 \right) \left( \mu_n + \mu_p \right)

   s_n = -R + ...
   
   s_p = -R + ...

