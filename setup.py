from distutils.core import setup
from Cython.Build import cythonize

setup(name='c_decision_stump',
      ext_modules=cythonize(['quadboost/weak_learner/c_decision_stump.pyx'], annotate=True))
