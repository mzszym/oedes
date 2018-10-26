Introduction
============

``oedes`` (Open Electronic DEvice Simulator) is a Python software package for modeling of electronic devices. It is primarily focused on emerging electronic devices. It is applicable to organic electronic, electrochemical, bioelectronic, and Peroskvite based devices.

`oedes` was written to take into account both the special aspects of non-conventional electronic devices, and modern trends in software development. The result a small but powerful package, which is tightly integrated with standard scientific software stack. This simplifies both running and sharing the simulations.

`oedes` mission is to:

- enable `open science` research in emerging electronic devices, by enabling full disclosure of numerical simulations;
- allow reuse of established device models, which were usually published without a runnanble implementation;
- accelerate research by providing a platform for parameter extraction from measurements and for device optimization driven by simulation;
- simplify development of new simulation models and tools, by providing a library of robust components;
- provide a standard way of running and sharing device simulations.

`oedes` is released as open-source under GNU Affero General Public License v3. It is free to use and distributed with complete source code. 

Conventions
-----------

Code which is intended to execute in system command-line starts with ``$``, for example:

.. code-block:: bash

   $ python

All code which is indendent to be run in Python starts with ``>>>`` or ``...``, for example:

.. code-block:: python

   >>> print('hello world')

Sequences of characters which can be copied into code in current context are emphasized as ``print`` or ``1+2+3``. ``oedes`` formatted in that way refers to name of Python software package, or to Python symbol in current example.

General concepts are emphasized as `params`. `oedes` written in that way refers to the software project.
