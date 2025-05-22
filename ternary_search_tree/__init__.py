"""
Ternary Search Tree Package

A comprehensive implementation of a Ternary Search Tree (TST) data structure in Python,
designed for efficient string storage, retrieval, and prefix matching operations.

Classes:
    TSTNode: Individual node in the ternary search tree
    TernarySearchTree: Main tree implementation with all operations

Example:
    >>> from ternary_search_tree import TernarySearchTree
    >>> tst = TernarySearchTree()
    >>> tst.insert("hello")
    >>> tst.insert("world")
    >>> tst.search("hello")
    True
    >>> tst.prefix_search("hel")
    ['hello']

Authors: [Your Names]
Version: 1.0.0
"""

from .node import TSTNode
from .tree import TernarySearchTree

# Package metadata
__version__ = "1.0.0"
__author__ = "[Your Names]"
__email__ = "[Your Email]"
__description__ = "Ternary Search Tree implementation for efficient string operations"

# Public API
__all__ = [
    'TSTNode',
    'TernarySearchTree',
]

# Version information
def get_version():
    """Return the package version."""
    return __version__

def get_info():
    """Return package information."""
    return {
        'name': 'ternary_search_tree',
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'classes': __all__
    }