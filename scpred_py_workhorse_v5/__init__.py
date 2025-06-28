# Makes 'scpred_py' a package and can expose key functions/classes.
# For now, we'll import our main class.

"""
scPred-Py: A Python implementation of the scPred single-cell classification tool.
"""

from ._core import ScPredModel

__version__ = "0.1.0"
__all__ = ["ScPredModel"]

print("scPred-Py initialized. Remember this is a starting point!")