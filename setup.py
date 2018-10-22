import numpy as np
from setuptools import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext


ext_modules = [
    Extension("graphbispectrum.simulortho", ["graphbispectrum/simulortho.pyx"]),
    Extension("graphbispectrum.csymmetric_group", ["graphbispectrum/csymmetric_group.pyx"]),
    Extension("graphbispectrum.function", ["graphbispectrum/function.pyx"]),
]


setup(
    name="graphbispectrum",
    version="0.0.1",
    description="",
    cmdclass={"build_ext": build_ext},
    include_dirs=[np.get_include()],
    ext_modules=ext_modules,
    zip_safe=False)
