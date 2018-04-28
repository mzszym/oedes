Installation
============

Requirements
------------

- Python 2.7 or Python 3.4+
- sparsegrad
- Python scientific stack

Installation from PyPI
----------------------

This is the preferred way of installing ``oedes``. It automatically takes care of all dependencies.

Two variants of the installation are possible:

- system wide installation:

.. code-block:: bash

   $ pip install oedes

- local installation not requiring administrator's rights:

.. code-block:: bash

   $ pip install oedes --user

In the case of local installation, ``oedes`` is installed inside user's home directory. In Linux, this defaults to ``$HOME/.local``.

Verifying the installation
--------------------------

After installing, it is advised to run the test suite to ensure that ``oedes`` works correctly on your system:

.. doctest::
   :options: +SKIP

   >>> import oedes
   >>> oedes.test()
   Running unit tests for oedes...
   OK
   <nose.result.TextTestResult run=15 errors=0 failures=0>

If any errors are found, ``oedes`` is not compatible with your system. Either your Python scientific stack is too old, or there is a bug. 

oedes is evolving, and backward compatibility is not yet offered. It is recommended to check which version you are using:

.. doctest::

   >>> import oedes
   >>> oedes.version
   '0.0.14'

Development installation (advanced)
-----------------------------------

Current development version of sparsegrad can be installed from the development repository by running

.. code-block:: bash

   $ git clone https://github.com/mzszym/oedes.git
   $ cd oedes
   $ pip install -e .

The option ``-e`` tells that ``oedes`` code should be loaded from ``git`` controlled directory, instead of being copied to the Python libraries directory. As with the regular installation, ``--user`` option should be appended for local installation.


