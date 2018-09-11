``oedes`` - organic electronic device simulator
===============================================

|Travis-CI-badge| |Readthedocs-badge| |Binder-badge|


This is work in progress. See ``doc/`` for documentation, and ``examples/`` for examples of use.

Installation
------------

.. code:: bash

   pip install oedes

It is recommended to run test suite after installing

.. code:: bash

   python -c "import oedes; oedes.test()"


Example simulation
------------------

This builds and solves a model of abrupt PN junction:

.. code:: python

   import oedes
   from oedes import models
   
   # Define doping profile
   def doping_profile(mesh, ctx, eq):
       Nd = ctx.param(eq, 'Nd')
       Na = ctx.param(eq,'Na')
       return oedes.ad.where(mesh.x<mesh.length*0.5,Nd,-Na)
   
   # Define device model
   poisson = models.PoissonEquation()
   temperature = models.ConstTemperature()
   electron = models.BandTransport(poisson=poisson, name='electron', z=-1, thermal=temperature)
   hole = models.BandTransport(poisson=poisson, name='hole', z=1, thermal=temperature)
   doping = models.FixedCharge(poisson, density=doping_profile)
   semiconductor = models.Electroneutrality([electron, hole, doping],name='semiconductor')
   recombination = models.DirectRecombination(semiconductor)
   anode = models.OhmicContact(poisson, semiconductor, 'electrode0')
   cathode = models.OhmicContact(poisson, semiconductor, 'electrode1')
   equations=[ poisson, temperature, electron, hole, 
               doping, semiconductor, anode, cathode,
               recombination ]
   
   # Define device parameters
   params={
       'T':300,
       'epsilon_r':12,
       'Na':1e24,
       'Nd':1e24,
       'hole.mu':1,
       'electron.mu':1,
       'hole.energy':-1.1,
       'electron.energy':0,
       'electrode0.voltage':0,
       'electrode1.voltage':0,
       'hole.N0':1e27,
       'electron.N0':1e27,
       'beta':1e-9
   }
    
   # Discretize and solve discrete model
   mesh = oedes.fvm.mesh1d(100e-9)
   model = oedes.fvm.discretize(equations, mesh)
   c=oedes.context(model)
   c.solve(params)
     
   # Plot bands and quasi Fermi potentials
   import matplotlib.pylab as plt
   p=c.mpl(plt.gcf(), plt.gca())
   p.plot(['electron.Eband'],label='$E_c$')
   p.plot(['hole.Eband'],label='$E_v$')
   p.plot(['electron.Ef'],linestyle='--',label='$E_{Fn}$')
   p.plot(['hole.Ef'],linestyle='-.',label='$E_{Fp}$')
   p.apply_settings({'xunit':'n','xlabel':'nm'})
   p.ax.legend(loc=0,frameon=False)
   plt.show()

.. image:: doc/fig/tutorial-pn.png
         
.. |Travis-CI-badge| image:: https://travis-ci.org/mzszym/oedes.svg?branch=master
   :target: https://travis-ci.org/mzszym/oedes

.. |Readthedocs-badge| image:: https://readthedocs.org/projects/oedes/badge/?version=latest
   :target: http://oedes.readthedocs.io/en/latest/?badge=latest
      
.. |Binder-badge| image:: https://mybinder.org/badge.svg
   :target: https://mybinder.org/v2/gh/mzszym/oedes.git/master
