import numpy
import setuptools
from Cython.Build import cythonize

setuptools.setup(
    name="cure",
    ext_modules=cythonize("cluster_utils.pyx"),
    zip_safe=False,
)