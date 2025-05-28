"""
Unit tests for the Ternary Search Tree implementation.

This module contains comprehensive tests for all TST operations including:
- Basic insertion and search
- Prefix searching
- Edge cases (empty strings, duplicates)
- Performance characteristics
"""

import unittest
import time
from ternary_search_tree import TernarySearchTree


class TestTernarySearchTree(unittest.TestCase):
    """Test cases for TernarySearchTree class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.tst = TernarySearchTree()
    
    def test_empty_tree(self):
        """Test operations on an empty tree."""
        self.assertEqual(len(self.tst), 0)
        self.assertFalse(self.tst.search("test"))
        self.assertFalse(self.tst.search(""))
        self.assertEqual(self.tst.all_strings(), [])
        self.assertEqual(self.tst.prefix_search("a"), [])
    
    def test_single_insertion(self):
        """Test inserting a single string."""
        self.tst.insert("test")
        self.assertEqual(len(self.tst), 1)
        self.assertTrue(self.tst.search("test", exact=True))
        self.assertTrue(self.tst.search("tes"))  # prefix
        self.assertFalse(self.tst.search("testing"))
        self.assertEqual(self.tst.all_strings(), ["test"])
    
    def test_empty_string_insertion(self):
        """Test inserting an empty string."""
        self.tst.insert("")
        self.assertEqual(len(self.tst), 1)
        self.assertTrue(self.tst.search("", exact=True))
        self.assertTrue(self.tst.search(""))
        self.assertEqual(self.tst.all_strings(), [""])
    
    def test_multiple_insertions(self):
        """Test inserting multiple strings."""
        words = ["cat", "car", "card", "care", "careful", "cats", "dog"]
        for word in words:
            self.tst.insert(word)
        
        self.assertEqual(len(self.tst), len(words))
        
        # Test exact searches
        for word in words:
            self.assertTrue(self.tst.search(word, exact=True))
        
        # Test prefix searches
        self.assertTrue(self.tst.search("ca"))
        self.assertTrue(self.tst.search("car"))
        self.assertTrue(self.tst.search("care"))
        
        # Test non-existent words
        self.assertFalse(self.tst.search("bat", exact=True))
        self.assertFalse(self.tst.search("careless", exact=True))
    
    def test_duplicate_insertions(self):
        """Test that duplicate insertions don't increase size."""
        self.tst.insert("test")
        self.tst.insert("test")
        self.tst.insert("test")
        
        self.assertEqual(len(self.tst), 1)
        self.assertTrue(self.tst.search("test", exact=True))
    
    def test_prefix_search(self):
        """Test prefix search functionality."""
        words = ["cat", "car", "card", "care", "careful", "cats", "dog", "door"]
        for word in words:
            self.tst.insert(word)
        
        # Test various prefixes
        ca_words = self.tst.prefix_search("ca")
        self.assertEqual(sorted(ca_words), ["car", "card", "care", "careful", "cat", "cats"])
        
        car_words = self.tst.prefix_search("car")
        self.assertEqual(sorted(car_words), ["car", "card", "care", "careful"])
        
        d_words = self.tst.prefix_search("d")
        self.assertEqual(sorted(d_words), ["dog", "door"])
        
        # Test non-existent prefix
        self.assertEqual(self.tst.prefix_search("xyz"), [])
        
        # Test empty prefix (should return all words)
        all_words = self.tst.prefix_search("")
        self.assertEqual(sorted(all_words), sorted(words))
    
    def test_all_strings(self):
        """Test retrieving all strings from the tree."""
        words = ["banana", "apple", "cherry", "date", "elderberry"]
        for word in words:
            self.tst.insert(word)
        
        all_strings = self.tst.all_strings()
        self.assertEqual(sorted(all_strings), sorted(words))
    
    def test_contains_operator(self):
        """Test the __contains__ operator (in keyword)."""
        words = ["hello", "world", "python"]
        for word in words:
            self.tst.insert(word)
        
        for word in words:
            self.assertIn(word, self.tst)
        
        self.assertNotIn("java", self.tst)
        self.assertNotIn("", self.tst)
    
    def test_iteration(self):
        """Test iteration over the tree."""
        words = ["zebra", "apple", "banana", "cherry"]
        for word in words:
            self.tst.insert(word)
        
        iterated_words = list(self.tst)
        self.assertEqual(iterated_words, sorted(words))
    
    def test_string_representation(self):
        """Test string representation of the tree."""
        self.tst.insert("abc")
        self.tst.insert("abd")
        
        tree_str = str(self.tst)
        self.assertIn("char: a", tree_str)
        self.assertIn("char: b", tree_str)
        self.assertIn("char: c", tree_str)
        self.assertIn("char: d", tree_str)
        self.assertIn("terminates:", tree_str)
    
    def test_edge_cases(self):
        """Test various edge cases."""
        # Single character strings
        self.tst.insert("a")
        self.tst.insert("z")
        self.assertEqual(len(self.tst), 2)
        self.assertTrue(self.tst.search("a", exact=True))
        self.assertTrue(self.tst.search("z", exact=True))
        
        # Clear tree for next test
        self.tst = TernarySearchTree()
        
        # Nested prefixes
        self.tst.insert("a")
        self.tst.insert("ab")
        self.tst.insert("abc")
        self.assertEqual(len(self.tst), 3)
        
        # All should be found as exact matches
        self.assertTrue(self.tst.search("a", exact=True))
        self.assertTrue(self.tst.search("ab", exact=True))
        self.assertTrue(self.tst.search("abc", exact=True))
    
    def test_case_sensitivity(self):
        """Test that the tree is case-sensitive."""
        self.tst.insert("Test")
        self.tst.insert("test")
        
        self.assertEqual(len(self.tst), 2)
        self.assertTrue(self.tst.search("Test", exact=True))
        self.assertTrue(self.tst.search("test", exact=True))
        self.assertFalse(self.tst.search("TEST", exact=True))
    
    def test_special_characters(self):
        """Test handling of special characters."""
        special_words = ["hello@world.com", "test_123", "word-with-dashes", "unicode_test_café"]
        
        for word in special_words:
            self.tst.insert(word)
        
        self.assertEqual(len(self.tst), len(special_words))
        
        for word in special_words:
            self.assertTrue(self.tst.search(word, exact=True))
    
    def test_tree_statistics(self):
        """Test tree statistics functionality."""
        # Empty tree
        stats = self.tst.get_stats()
        self.assertEqual(stats['size'], 0)
        self.assertEqual(stats['nodes'], 0)
        
        # Non-empty tree
        words = ["cat", "car", "dog"]
        for word in words:
            self.tst.insert(word)
        
        stats = self.tst.get_stats()
        self.assertEqual(stats['size'], 3)
        self.assertGreater(stats['nodes'], 0)
        self.assertGreater(stats['height'], 0)
        self.assertGreater(stats['avg_depth'], 0)


class TestTSTPerformance(unittest.TestCase):
    """Performance tests for TST operations."""
    
    def test_insertion_performance(self):
        """Test insertion performance with a large number of words."""
        tst = TernarySearchTree()
        
        # Generate test words
        import string
        words = []
        for i in range(1000):
            word = ''.join([chr(ord('a') + (i + j) % 26) for j in range(5)])
            words.append(word)
        
        # Time insertions
        start_time = time.time()
        for word in words:
            tst.insert(word)
        insertion_time = time.time() - start_time
        
        self.assertLess(insertion_time, 5.0)  # Should complete in under 5 seconds
        self.assertEqual(len(tst), len(set(words)))  # Handle duplicates
    
    def test_search_performance(self):
        """Test search performance."""
        tst = TernarySearchTree()
        
        # Insert many words
        words = [f"word{i:04d}" for i in range(1000)]
        for word in words:
            tst.insert(word)
        
        # Time searches
        start_time = time.time()
        for word in words[:100]:  # Search for first 100 words
            self.assertTrue(tst.search(word, exact=True))
        search_time = time.time() - start_time
        
        self.assertLess(search_time, 1.0)  # Should complete quickly


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)