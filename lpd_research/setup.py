#from distutils.core import setup
from setuptools import setup
#from distutils.extension import Extension
#from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np
#default_includes = [numpy.get_include(), ".", "lpresearch"]
#ext_modules = [Extension("lpresearch.cspdecoder", ["lpresearch/cspdecoder.pyx"], include_dirs=default_includes)]

cythonModules = ['lpresearch.cspdecoder', 'pseudoweight.supportsearch']
import os.path
def makeExtensions():
    """Returns an Extension object for the given submodule of lpdecoding."""
    sources = [module.replace(".", os.path.sep) + ".pyx" for module in cythonModules ]
    cyt = cythonize(sources)
    for c in cyt:
        c.include_dirs = [np.get_include(), "."]
    return cyt
setup(
  name = 'LP Decoding Research Modules',
  packages = ["lpresearch"],
  test_suite = 'tests',
  ext_modules=makeExtensions()
)
