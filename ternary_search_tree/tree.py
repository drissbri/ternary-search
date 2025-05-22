"""
Ternary Search Tree implementation.

This module contains the main TernarySearchTree class which provides
all the functionality for string storage, retrieval, and manipulation.
"""

from typing import Optional, List, Iterator
from .node import TSTNode


class TernarySearchTree:
    """
    Ternary Search Tree implementation for efficient string storage and retrieval.
    
    Time Complexity:
    - Insert: O(log n + k) average, O(n + k) worst case
    - Search: O(log n + k) average, O(n + k) worst case
    - Delete: O(log n + k) average, O(n + k) worst case
    
    Where n is the number of strings and k is the length of the string.
    
    Space Complexity: O(n * k) where n is the number of strings and k is average string length
    """
    
    def __init__(self):
        """Initialize an empty ternary search tree."""
        self.root: Optional[TSTNode] = None
        self._size = 0
    
    def insert(self, string: str) -> None:
        """
        Insert a string into the ternary search tree.
        
        Args:
            string: The string to insert
            
        Time Complexity: O(log n + k) average case, O(n + k) worst case
        """
        if string == "":
            # Handle empty string - create root if it doesn't exist
            if self.root is None:
                self.root = TSTNode(None)  # Root node for empty string has None as char
            if not self.root.is_end_of_word:
                self.root.is_end_of_word = True
                self._size += 1
            return
        
        # For non-empty strings, ensure root exists
        if self.root is None:
            self.root = TSTNode(None)  # Initialize root with None char
        
        # Insert the string starting from root's equal child
        self.root.equal = self._insert_recursive(self.root.equal, string, 0)
    
    def _insert_recursive(self, node: Optional[TSTNode], string: str, index: int) -> TSTNode:
        """
        Recursively insert a string into the TST.
        
        Args:
            node: Current node
            string: String to insert
            index: Current character index
            
        Returns:
            The node (possibly newly created)
        """
        char = string[index]
        
        if node is None:
            node = TSTNode(char)
        
        if char < node.char:
            node.left = self._insert_recursive(node.left, string, index)
        elif char > node.char:
            node.right = self._insert_recursive(node.right, string, index)
        else:
            # char == node.char
            if index < len(string) - 1:
                node.equal = self._insert_recursive(node.equal, string, index + 1)
            else:
                # End of string
                if not node.is_end_of_word:
                    node.is_end_of_word = True
                    self._size += 1
        
        return node
    
    def search(self, string: str, exact: bool = False) -> bool:
        """
        Search for a string in the ternary search tree.
        
        Args:
            string: The string to search for
            exact: If True, only return True if the string is a complete word
                  If False, return True if the string is a prefix of any word
                  
        Returns:
            True if string is found (as word or prefix), False otherwise
            
        Time Complexity: O(log n + k) average case, O(n + k) worst case
        """
        if string == "":
            # Empty string handling
            if exact:
                return self.root is not None and self.root.is_end_of_word
            else:
                return self.root is not None
        
        # For non-empty strings, search starting from root's equal child
        if self.root is None:
            return False
            
        node = self._search_node_from(self.root.equal, string, 0)
        
        if node is None:
            return False
        
        if exact:
            return node.is_end_of_word
        else:
            return True
    
    def _search_node_from(self, node: Optional[TSTNode], string: str, index: int) -> Optional[TSTNode]:
        """
        Find the node corresponding to the end of the given string starting from a specific node.
        
        Args:
            node: Starting node
            string: String to search for
            index: Current character index
            
        Returns:
            The node at the end of the string path, or None if not found
        """
        if node is None or index >= len(string):
            return node if index == len(string) else None
        
        char = string[index]
        
        while node is not None:
            if char < node.char:
                node = node.left
            elif char > node.char:
                node = node.right
            else:
                # char == node.char
                if index == len(string) - 1:
                    return node
                return self._search_node_from(node.equal, string, index + 1)
        
        return None
    
    def _search_node(self, string: str) -> Optional[TSTNode]:
        """
        Find the node corresponding to the end of the given string.
        
        Args:
            string: String to search for
            
        Returns:
            The node at the end of the string path, or None if not found
        """
        if string == "":
            return self.root
        
        if self.root is None:
            return None
            
        return self._search_node_from(self.root.equal, string, 0)
    
    def prefix_search(self, prefix: str) -> List[str]:
        """
        Find all strings that start with the given prefix.
        
        Args:
            prefix: The prefix to search for
            
        Returns:
            List of all strings with the given prefix
            
        Time Complexity: O(log n + k + m) where m is the number of results
        """
        if prefix == "":
            return self.all_strings()
        
        if self.root is None:
            return []
        
        node = self._search_node_from(self.root.equal, prefix, 0)
        if node is None:
            return []
        
        results = []
        
        # If the prefix itself is a word, include it
        if node.is_end_of_word:
            results.append(prefix)
        
        # Find all words that extend this prefix
        self._collect_words(node.equal, prefix, results)
        
        return sorted(results)
    
    def _collect_words(self, node: Optional[TSTNode], current_string: str, results: List[str]) -> None:
        """
        Recursively collect all words from a subtree.
        
        Args:
            node: Current node
            current_string: String built so far
            results: List to collect results
        """
        if node is None:
            return
        
        # Traverse left subtree
        self._collect_words(node.left, current_string, results)
        
        # Process current node
        new_string = current_string + node.char
        if node.is_end_of_word:
            results.append(new_string)
        
        # Traverse equal subtree (continue building the string)
        self._collect_words(node.equal, new_string, results)
        
        # Traverse right subtree
        self._collect_words(node.right, current_string, results)
    
    def all_strings(self) -> List[str]:
        """
        Return all strings stored in the ternary search tree.
        
        Returns:
            Sorted list of all strings in the tree
            
        Time Complexity: O(n * k) where n is number of strings and k is average length
        """
        results = []
        
        if self.root is None:
            return results
        
        # Check if empty string is stored (root node marks end of word)
        if self.root.is_end_of_word:
            results.append("")
        
        # Collect all non-empty strings starting from root's equal child
        if self.root.equal is not None:
            self._collect_words(self.root.equal, "", results)
        
        return sorted(results)
    
    def __len__(self) -> int:
        """Return the number of strings stored in the tree."""
        return self._size
    
    def __contains__(self, string: str) -> bool:
        """Check if a string is in the tree (exact match)."""
        return self.search(string, exact=True)
    
    def __iter__(self) -> Iterator[str]:
        """Iterate over all strings in the tree in sorted order."""
        return iter(self.all_strings())
    
    def __str__(self) -> str:
        """Return a string representation of the tree."""
        if self.root is None:
            return "Empty TernarySearchTree"
        
        lines = []
        self._build_string_representation(self.root, "", "", lines)
        return "\n".join(lines)
    
    def _build_string_representation(self, node: Optional[TSTNode], prefix: str, 
                                   edge_type: str, lines: List[str]) -> None:
        """
        Build a string representation of the tree structure.
        
        Args:
            node: Current node
            prefix: Prefix for indentation
            edge_type: Type of edge ("", "_lt:", "_eq:", "_gt:")
            lines: List to collect output lines
        """
        if node is None:
            return
        
        if node.char is None:
            # Root node
            lines.append(f"ROOT (empty string): terminates={node.is_end_of_word}")
        else:
            lines.append(f"{prefix}{edge_type}char: '{node.char}', terminates: {node.is_end_of_word}")
        
        # Calculate new prefix for children
        if node.char is None:
            new_prefix = "  "
        else:
            new_prefix = prefix + "  "
        
        # Add children (only show non-None children)
        if node.left is not None:
            self._build_string_representation(node.left, new_prefix, "_lt:", lines)
        
        if node.equal is not None:
            self._build_string_representation(node.equal, new_prefix, "_eq:", lines)
        
        if node.right is not None:
            self._build_string_representation(node.right, new_prefix, "_gt:", lines)
    
    def get_stats(self) -> dict:
        """
        Get statistics about the tree.
        
        Returns:
            Dictionary with tree statistics
        """
        if self.root is None:
            return {
                'size': 0,
                'height': 0,
                'nodes': 0,
                'avg_depth': 0
            }
        
        total_depth = 0
        node_count = 0
        max_depth = 0
        
        def calculate_stats(node: Optional[TSTNode], depth: int) -> None:
            nonlocal total_depth, node_count, max_depth
            
            if node is None:
                return
            
            node_count += 1
            total_depth += depth
            max_depth = max(max_depth, depth)
            
            calculate_stats(node.left, depth + 1)
            calculate_stats(node.equal, depth + 1)
            calculate_stats(node.right, depth + 1)
        
        calculate_stats(self.root, 0)
        
        return {
            'size': self._size,
            'height': max_depth,
            'nodes': node_count,
            'avg_depth': total_depth / node_count if node_count > 0 else 0
        }