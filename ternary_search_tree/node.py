"""
TST Node implementation.

This module contains the TSTNode class which represents individual nodes
in a Ternary Search Tree.
"""

from typing import Optional


class TSTNode:
    """
    A node in a Ternary Search Tree.
    
    Each node contains:
    - A character (can be None for root)
    - A boolean flag indicating if this node marks the end of a word
    - Three child pointers: left (less than), equal, right (greater than)
    """
    
    def __init__(self, char: Optional[str] = None):
        """
        Initialize a TST node.
        
        Args:
            char: The character stored in this node
        """
        self.char = char
        self.is_end_of_word = False
        self.left: Optional['TSTNode'] = None
        self.equal: Optional['TSTNode'] = None
        self.right: Optional['TSTNode'] = None
    
    def __repr__(self):
        """Return string representation of the node."""
        return f"TSTNode(char='{self.char}', is_end={self.is_end_of_word})"
    
    def __str__(self):
        """Return human-readable string representation."""
        char_display = self.char if self.char is not None else 'None'
        return f"Node('{char_display}', end={self.is_end_of_word})"