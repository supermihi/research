#from distutils.core import setup
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
default_includes = [numpy.get_include(), ".", "src"]
ext_modules = [Extension("cspdecoder", ["cspdecoder.pyx"], include_dirs=default_includes)]

setup(
  name = 'Constrained Shortest Path Decoder',
  cmdclass = {'build_ext': build_ext},
  packages = ["lpresearch"],
  ext_modules = ext_modules
)
