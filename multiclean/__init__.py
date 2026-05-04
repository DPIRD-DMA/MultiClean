from importlib.metadata import version

from .multiclean import clean_array

__version__ = version("multiclean")

__all__ = ["clean_array", "__version__"]
