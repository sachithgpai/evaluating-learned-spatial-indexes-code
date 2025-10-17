# cython_setup.py

from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        "qd_tree_query_cost_calculator.pyx", compiler_directives={"language_level": "3"},annotate=True
    )
)