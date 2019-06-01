from distutils.core import setup, Extension
from Cython.Build import cythonize

import numpy

extensions = []

extensions.append(Extension("test_by_mark", ["test_by_mark.pyx", "model_IHC_BEZ2018.c", "complex.c"]))

setup(ext_modules=cythonize(extensions), include_dirs=[numpy.get_include()])
