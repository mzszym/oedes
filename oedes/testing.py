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

import base64
import json
import numpy as np
from contextlib import contextmanager
import os
import fnmatch
import subprocess
import tempfile
import time
import functools

array_hooks = []


@contextmanager
def printall():
    opt = np.get_printoptions()
    np.set_printoptions(threshold=np.inf)
    yield
    np.set_printoptions(**opt)


def array2dict(obj):
    assert isinstance(obj, np.ndarray)
    obj_data = np.ascontiguousarray(obj)
    return dict(__ndarray__=base64.b64encode(obj_data).decode('ascii'),
                dtype=str(obj.dtype),
                shape=",".join(map(str, obj.shape)))


def isarray(obj):
    return isinstance(obj, dict) and '__ndarray__' in obj


def dict2array(dct):
    assert isarray(dct)
    data = base64.b64decode(dct['__ndarray__'])
    shape = [int(s) for s in dct['shape'].split(',')]
    return np.frombuffer(data, np.dtype(dct['dtype'])).reshape(shape)


def nb_store_array(x, atol=0., rtol=1e-7, label=None):
    for hook in array_hooks:
        if hook(x, atol=atol, rtol=rtol, label=label):
            return
    x = np.asarray(x)
    if x.dtype == np.longdouble:
        x = np.asarray(x, dtype=np.double)
    elif x.dtype == np.longcomplex:
        x = np.asarray(x, dtype=np.complex)
    try:
        from IPython.display import publish_display_data
    except BaseException:
        return
    d = array2dict(x)
    u = dict2array(d)
    assert (x == u).all()
    data = {'application/json': dict(data=array2dict(x))}
    metadata = dict(format='oedes.testing.nb_store_array',
                    atol=atol, rtol=rtol, label=label)
    publish_display_data(data=data, metadata=metadata)


store = nb_store_array


class _store_result(object):
    def __init__(self, func, **kwargs):
        self.func = func
        self.store_kwargs = kwargs

    def __call__(self, *args, **kwargs):
        result = self.func(*args, **kwargs)
        nb_store_array(result, **self.store_kwargs)
        return result


def stored(**kwargs):
    return functools.partial(_store_result, **kwargs)


def stored_arrays(cells):
    for cell in cells:
        if cell.cell_type != 'code':
            continue
        for output in cell.outputs:
            try:
                json = output.data['application/json'].data
                meta = output.metadata
                if meta.format != 'oedes.testing.nb_store_array' or not isarray(
                        json):
                    continue
            except KeyError:
                continue
            except AttributeError:
                continue
            yield dict((k, v) for k, v in meta.items() if k != 'format'), dict2array(json)


def check_notebook_arrays(reference, tested):
    ro = list(stored_arrays(reference.cells))
    to = list(stored_arrays(tested.cells))
    if len(ro) != len(to):
        raise RuntimeError(
            'different number of arrays between reference and tested')
    for iarr, (ref, test) in enumerate(zip(ro, to)):
        rmeta, rarr = ref
        tmeta, tarr = test
        if not np.allclose(rarr, tarr, atol=rmeta['atol'], rtol=rmeta['rtol']):
            with printall():
                raise RuntimeError('\n'.join(
                    ["Difference in stored array %d" % iarr,
                     "Maximum error is %e" % np.amax(np.abs(rarr - tarr)),
                     "Maximum relative error is %e" % np.amax(
                         np.abs((rarr - tarr) / rarr)),
                     "Assumed rtol=%e atol=%e" % (
                         rmeta['rtol'], rmeta['atol']),
                     "Reference:",
                     str(rarr),
                     "Tested:",
                     str(tarr)
                     ]))
    return True


def read_notebook(fn):
    from nbformat.converter import convert
    from nbformat.reader import reads
    IPYTHON_VERSION = 4
    NBFORMAT_VERSION = 4

    with open(fn, 'r') as f:
        text = f.read()
        data = reads(text)
    return text, convert(data, NBFORMAT_VERSION)


def run_notebook(fn, timeout):
    _, reference = read_notebook(fn)
    temporary = tempfile.NamedTemporaryFile(suffix='.ipynb', delete=False)
    temporary.close()
    try:
        tempdir, tempname = os.path.split(temporary.name)
        for attempt in range(10):
            process = subprocess.Popen(['jupyter', 'nbconvert', '--ExecutePreprocessor.timeout=%d' % timeout, '--ExecutePreprocessor.kernel_name=python',
                                        '--to', 'notebook', '--execute', fn, '--output', tempname, '--output-dir', tempdir], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            if stdout is not None:
                stdout = stdout.decode()
            if stderr is not None:
                stderr = stderr.decode()
                if stderr.find('ZMQError: Address already in use') >= 0:
                    time.sleep(np.random.rand() + 0.1)
                    continue
            if process.returncode != 0:
                raise RuntimeError("\n".join(['Unable to execute notebook (returned exitcode %d)' % process.returncode,
                                              'Output:',
                                              stdout,
                                              'Errors:',
                                              stderr]))
            text, tested = read_notebook(temporary.name)
            return reference, tested, text
        raise RuntimeError(
            'workaround: maximum number of %d attempts reached' % attempt)
    finally:
        os.remove(temporary.name)

# def run_notebook(fn, **kwargs):
#    import nbformat
#    from nbconvert.preprocessors import ExecutePreprocessor
#    import zmq
#    with open(fn,'r') as f:
#        ref=nbformat.read(f,as_version=4)
#    for attempt in range(10):
#        try:
#            ep=ExecutePreprocessor(kernel_name='python',**kwargs)
#            result,res=ep.preprocess(ref,{})
#            break
#        except zmq.ZMQError:
#            continue
#    return ref,result,nbformat.writes(result)


def find_notebooks(rootdir='.'):
    for root, dirnames, filenames in os.walk(rootdir):
        root = os.path.relpath(root, rootdir)
        if any([x.startswith('.') for x in os.path.split(root)]):
            continue
        for filename in fnmatch.filter(filenames, '*.ipynb'):
            f = os.path.join(root, filename)
            if os.path.islink(f):
                continue
            yield os.path.relpath(f, rootdir)


from oedes.fvm import mesh1d
from oedes.models import electronic_device, BaseModel


def unipolar(L, polarity='p'):
    mesh = mesh1d(L)
    model = BaseModel()
    electronic_device(model, mesh, polarity)
    prefix = model.species[0].name
    params = {
        'T': 300,
        'epsilon_r': 3.,
        'electrode0.voltage': 0,
        'electrode1.voltage': 0,
        'electrode0.workfunction': 0,
        'electrode1.workfunction': 0,
        prefix + '.energy': 0,
        prefix + '.N0': 1e27,
        prefix + '.mu': 1e-9
    }
    return model, params


from oedes.fvm import multilayer1d
from oedes.models import CompositeModel, Equal, FermiLevelEqual, FermiLevelEqualElectrode, AppliedVoltage


def bilayer(L0, L1, polarity='p'):
    ml = multilayer1d([('layer0', mesh1d(L0, boundary_names=['electrode0', 'internal'])),
                       ('layer1', mesh1d(L1, boundary_names=['internal', 'electrode1']))])
    model = CompositeModel()
    params = {
        'layer0.electrode0.voltage': 0,
        'layer1.electrode1.voltage': 0,
        'layer0.electrode0.workfunction': 0,
        'layer1.electrode1.workfunction': 0
    }
    if 'n' in polarity and 'p' in polarity:
        params.update({
            'layer0.npi': 0,
            'layer1.npi': 0
        })
    for i, name in enumerate(ml.domains):
        sub = BaseModel()
        sub.name = name
        model.sub.append(sub)
        electronic_device(sub, ml.domains[name], polarity)
        ename = name.replace('layer', 'electrode')
        sub.poisson.bc = [AppliedVoltage(ename)]
        params.update({name + '.T': 300,
                       name + '.epsilon_r': 3.})
        for s in sub.species:
            s.bc = [FermiLevelEqualElectrode(ename)]
            prefix = name + '.' + s.prefix
            params.update({
                prefix + '.energy': 0,
                prefix + '.N0': 1e27,
                prefix + '.mu': 1e-9
            })
    l0, l1 = model.sub
    for i, s in enumerate(l1.species):
        s.bc.append(FermiLevelEqual(l0.species[i], 'internal'))
    l1.poisson.bc.append(Equal(l0.poisson, 'internal'))
    return model, params
