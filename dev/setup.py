from distutils.core import setup, Extension
from Cython.Build import cythonize

import numpy

include_dirs = []
include_dirs.append(numpy.get_include())

extensions = []
extensions.append(Extension("test_by_mark", ["test_by_mark.pyx", "complex.c", "model_IHC_BEZ2018.c", "model_Synapse_BEZ2018.c"]))

setup(ext_modules=cythonize(extensions), include_dirs=include_dirs)
