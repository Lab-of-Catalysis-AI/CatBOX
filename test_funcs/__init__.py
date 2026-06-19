from .base import TestFunction

try:
    from .pest import PestControl
except Exception:
    PestControl = None

try:
    from .MaxSAT.maximum_satisfiability import *
except Exception:
    pass

# Try to import Snar (requires Summit)
# try:
#     from .snar import Snar
# except ImportError:
#     pass
