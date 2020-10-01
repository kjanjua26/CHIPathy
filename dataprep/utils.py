"""
Helper functions
"""

import os


def remove_ext(path):
    """
    Returns filename by removing parent directory and extension from `path`
    """

    return os.path.splitext(os.path.basename(path))[0]


def roundmultiple(x, base):
    """
    Rounds X to the nearest multiple of BASE.
    """

    return base * round(x/base)