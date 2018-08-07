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

from setuptools import setup, Command

with open('oedes/_version.py') as version_file:
    exec(version_file.read())


class build_doc(Command):
    description = "build the documentation"

    user_options = []

    def run(self):
        import sphinx
        sphinx.main(['sphinx-build', '-b', 'html', 'doc', 'doc/_build/html'])

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass


class run_notebooks(Command):
    description = 'run notebooks'

    user_options = []

    def run(self):
        import fnmatch
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor
        import os.path
        import os
        all_ok = True
        for root, dirnames, filenames in os.walk('.'):
            if os.path.split(root)[-1] == '.ipynb_checkpoints':
                continue
            for name in fnmatch.filter(filenames, '*.ipynb'):
                f = os.path.join(root, name)
                print(f)
                try:
                    nb = nbformat.read(f, as_version=nbformat.NO_CONVERT)
                    ep = ExecutePreprocessor(timeout=600)
                    ep.preprocess(
                        nb, {'metadata': {'path': os.path.abspath(os.path.split(f)[0])}})
                    del ep
                except BaseException:
                    print('failed to execute %s' % f)
                    all_ok = False
                    continue
                nbformat.write(nb, f)
        assert all_ok

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass


setup(name='oedes',
      version=version,
      packages=['oedes',
                'oedes.models',
                'oedes.models.tests',
                'oedes.models.equations',
                'oedes.models.source',
                'oedes.functions',
                'oedes.functions.tests',
                'oedes.fvm',
                'oedes.optical',
                'oedes.optical.databases',
                'oedes.optical.test_materials',
                'oedes.optical.spectra',
                'oedes.optical.tests',
                'oedes.tests',
                'oedes.mpl',
                'oedes.utils'
                ],
      package_data={'oedes.optical.test_materials': ['*.yml'],
                    'oedes.optical.spectra': ['*.zip']},
      url='https://oedes.org/',
      license='GNU Affero General Public License v3',
      author='Marek Zdzislaw Szymanski',
      install_requires=[
          'numpy',
          'scipy',
          'sparsegrad>=0.0.9',
          'tqdm>=4.19',
          'nose',
          'matplotlib',
          'pyyaml',
          'six'
      ],
      author_email='marek@marekszymanski.com',
      description='organic electronic device simulator',
      long_description=open('README.rst').read(),
      include_package_data=True,
      platforms='any',
      classifiers=[
          'Programming Language :: Python',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 3',
          'Development Status :: 3 - Alpha',
          'Natural Language :: English',
          'Operating System :: OS Independent',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU Affero General Public License v3',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Scientific/Engineering :: Physics',
          'Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)'
      ],
      cmdclass={
          'build_doc': build_doc,
          'run_notebooks': run_notebooks
      }
      )
