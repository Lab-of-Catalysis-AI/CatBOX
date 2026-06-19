from .synthetic import *

try:
    from .xgboost_hp import *
except Exception:
    pass

try:
    from .Chemistry.chemistry import Chemistry as OCM
    Chemistry = OCM
except Exception:
    OCM = None
    Chemistry = None

try:
    from .DAR.DAR import DAR
except Exception:
    DAR = None

try:
    from .SCR.SCR import SCR
except Exception:
    SCR = None

from .benchmark_functions import Ackley as Ackley_benchmark, Rosenbrock as Rosenbrock_benchmark, Schwefel as Schwefel_benchmark, Griewank as Griewank_benchmark
