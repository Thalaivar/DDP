import setuptools
from Cython.Build import cythonize

setuptools.setup(
    name="CURE",
    version="0.0.1",
    author="Dhruv Laad",
    author_email="dhruvlaad@outlook.vom",
    description="An implementation of the CURE algorithm in pure python",
    url="https://github.com/Thalaivar/cure",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    ext_modules=cythonize("cluster_utils.pyx"),
    zip_safe=False,
)