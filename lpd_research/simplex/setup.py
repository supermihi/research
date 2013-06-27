from setuptools import setup
from Cython.Build import cythonize
import numpy as np

cyt = cythonize(["fixedpoint.pyx"])
for c in cyt:
    c.include_dirs = [np.get_include()]
setup(
  name = 'fixed point emulation',
  ext_modules=cyt
)



