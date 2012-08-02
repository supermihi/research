#from distutils.core import setup
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
default_includes = [numpy.get_include(), ".", "lpresearch"]
ext_modules = [Extension("lpresearch.cspdecoder", ["lpresearch/cspdecoder.pyx"], include_dirs=default_includes)]

setup(
  name = 'LP Decoding Research Modules',
  cmdclass = {'build_ext': build_ext},
  packages = ["lpresearch"],
  test_suite = 'tests',
  ext_modules = ext_modules
)
