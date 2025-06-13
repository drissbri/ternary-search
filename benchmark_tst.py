"""
Benchmarking script for Ternary Search Tree performance analysis.

This script measures and plots the performance of TST operations with increasing data sizes.
It's designed to be run on HPC infrastructure for comprehensive performance analysis.

Usage:
    python benchmark_tst.py [--output-dir OUTPUT_DIR] [--max-size MAX_SIZE]
"""

import time
import argparse
import random
import string
import json
import os
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np
from ternary_search_tree import TernarySearchTree


class TSTBenchmark:
    """Benchmark suite for Ternary Search Tree operations."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        """
        Initialize benchmark suite.
        
        Args:
            output_dir: Directory to save results and plots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Results storage
        self.results = {
            'insertion': {'sizes': [], 'times': [], 'avg_times': []},
            'search': {'sizes': [], 'times': [], 'avg_times': []},
            'prefix_search': {'sizes': [], 'times': [], 'avg_times': []},
            'memory': {'sizes': [], 'node_counts': []}
        }
    
    def generate_random_words(self, count: int, min_length: int = 3, max_length: int = 10) -> List[str]:
        """
        Generate random words for testing.
        
        Args:
            count: Number of words to generate
            min_length: Minimum word length
            max_length: Maximum word length
            
        Returns:
            List of random words
        """
        words = set()
        while len(words) < count:
            length = random.randint(min_length, max_length)
            word = ''.join(random.choices(string.ascii_lowercase, k=length))
            words.add(word)
        return list(words)
    
    def generate_prefix_words(self, count: int, common_prefixes: List[str]) -> List[str]:
        """
        Generate words with common prefixes for realistic testing.
        
        Args:
            count: Number of words to generate
            common_prefixes: List of common prefixes to use
            
        Returns:
            List of words with common prefixes
        """
        words = set()
        while len(words) < count:
            prefix = random.choice(common_prefixes)
            suffix_length = random.randint(1, 8)
            suffix = ''.join(random.choices(string.ascii_lowercase, k=suffix_length))
            words.add(prefix + suffix)
        return list(words)
    
    def benchmark_insertion(self, sizes: List[int]) -> None:
        """
        Benchmark insertion performance.
        
        Args:
            sizes: List of data sizes to test
        """
        print("Benchmarking insertion performance...")
        
        for size in sizes:
            print(f"  Testing insertion with {size} words...")
            
            # Generate test data
            words = self.generate_random_words(size)
            
            # Best case: balanced insertion order
            balanced_words = sorted(words)
            random.shuffle(balanced_words)  # Randomize for better balance
            
            # Worst case: sorted insertion order
            worst_case_words = sorted(words)
            
            # Average case: random insertion order
            avg_case_words = words.copy()
            random.shuffle(avg_case_words)
            
            # Test each case
            for case_name, test_words in [
                ("balanced", balanced_words),
                ("worst", worst_case_words),
                ("average", avg_case_words)
            ]:
                tst = TernarySearchTree()
                
                start_time = time.perf_counter()
                for word in test_words:
                    tst.insert(word)
                end_time = time.perf_counter()
                
                total_time = end_time - start_time
                avg_time = total_time / len(test_words)
                
                self.results['insertion']['sizes'].append(size)
                self.results['insertion']['times'].append(total_time)
                self.results['insertion']['avg_times'].append(avg_time)
                
                print(f"    {case_name} case: {total_time:.4f}s total, {avg_time*1000:.4f}ms avg")
    
    def benchmark_search(self, sizes: List[int]) -> None:
        """
        Benchmark search performance.
        
        Args:
            sizes: List of data sizes to test
        """
        print("Benchmarking search performance...")
        
        for size in sizes:
            print(f"  Testing search with {size} words...")
            
            # Generate and insert test data
            words = self.generate_random_words(size)
            tst = TernarySearchTree()
            
            for word in words:
                tst.insert(word)
            
            # Test search performance - scale search count with size but cap at reasonable limit
            search_count = min(1000, max(100, size // 50))  # More searches for larger datasets
            search_words = random.sample(words, min(search_count, len(words)))
            
            start_time = time.perf_counter()
            for word in search_words:
                result = tst.search(word, exact=True)
                assert result, f"Word {word} should be found"
            end_time = time.perf_counter()
            
            total_time = end_time - start_time
            avg_time = total_time / len(search_words)
            
            self.results['search']['sizes'].append(size)
            self.results['search']['times'].append(total_time)
            self.results['search']['avg_times'].append(avg_time)
            
            print(f"    {total_time:.4f}s total, {avg_time*1000:.4f}ms avg ({len(search_words)} searches)")
    
    def benchmark_prefix_search(self, sizes: List[int]) -> None:
        """
        Benchmark prefix search performance.
        
        Args:
            sizes: List of data sizes to test
        """
        print("Benchmarking prefix search performance...")
        
        common_prefixes = ["pre", "pro", "con", "com", "int", "exp", "imp", "out"]
        
        for size in sizes:
            print(f"  Testing prefix search with {size} words...")
            
            # Generate words with common prefixes
            words = self.generate_prefix_words(size, common_prefixes)
            tst = TernarySearchTree()
            
            for word in words:
                tst.insert(word)
            
            # Test prefix search performance - use fewer prefixes for very large datasets
            test_prefix_count = min(10, max(5, 20000 // size))  # Fewer tests for larger sizes
            test_prefixes = random.sample(common_prefixes, min(test_prefix_count, len(common_prefixes)))
            
            start_time = time.perf_counter()
            total_results = 0
            for prefix in test_prefixes:
                results = tst.prefix_search(prefix)
                total_results += len(results)
            end_time = time.perf_counter()
            
            total_time = end_time - start_time
            avg_time = total_time / len(test_prefixes)
            
            self.results['prefix_search']['sizes'].append(size)
            self.results['prefix_search']['times'].append(total_time)
            self.results['prefix_search']['avg_times'].append(avg_time)
            
            print(f"    {total_time:.4f}s total, {avg_time*1000:.4f}ms avg, {total_results} results")
    
    def benchmark_memory_usage(self, sizes: List[int]) -> None:
        """
        Benchmark memory usage (node count as proxy).
        
        Args:
            sizes: List of data sizes to test
        """
        print("Benchmarking memory usage...")
        
        for size in sizes:
            print(f"  Testing memory usage with {size} words...")
            
            words = self.generate_random_words(size)
            tst = TernarySearchTree()
            
            for word in words:
                tst.insert(word)
            
            stats = tst.get_stats()
            node_count = stats['nodes']
            
            self.results['memory']['sizes'].append(size)
            self.results['memory']['node_counts'].append(node_count)
            
            print(f"    {node_count} nodes, {node_count/size:.2f} nodes per word")
    
    def run_comprehensive_benchmark(self, max_size: int = 50000) -> None:
        """
        Run comprehensive benchmark suite.
        
        Args:
            max_size: Maximum data size to test
        """
        # Define test sizes with better scaling for large datasets
        if max_size <= 10000:
            sizes = [10, 50, 100, 500, 1000, 2000, 5000, 7500, 10000]
            sizes = [s for s in sizes if s <= max_size]
        else:
            # Extended range for larger datasets
            sizes = [10, 50, 100, 500, 1000, 2000, 5000, 10000, 15000, 25000, 35000, 50000]
            sizes = [s for s in sizes if s <= max_size]
        
        print(f"Running comprehensive benchmark with sizes: {sizes}")
        print(f"Maximum size: {max_size}")
        
        # Run all benchmarks
        self.benchmark_insertion(sizes)
        self.benchmark_search(sizes)
        self.benchmark_prefix_search(sizes)
        self.benchmark_memory_usage(sizes)
        
        # Save results
        self.save_results()
        
        # Generate plots
        self.plot_results()
    
    def save_results(self) -> None:
        """Save benchmark results to JSON file."""
        results_file = os.path.join(self.output_dir, "benchmark_results.json")
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {results_file}")
    
    def plot_results(self) -> None:
        """Generate performance plots."""
        print("Generating performance plots...")
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Ternary Search Tree Performance Analysis (Up to 50K Words)', fontsize=16)
        
        # Plot 1: Insertion time vs size
        if self.results['insertion']['sizes']:
            # Group by every 3 entries (balanced, worst, average cases)
            sizes = self.results['insertion']['sizes'][::3]
            balanced_times = self.results['insertion']['times'][::3]
            worst_times = self.results['insertion']['times'][1::3]
            avg_times = self.results['insertion']['times'][2::3]
            
            ax1.plot(sizes, balanced_times, 'g-o', label='Balanced case', markersize=4)
            ax1.plot(sizes[:len(worst_times)], worst_times, 'r-s', label='Worst case', markersize=4)
            ax1.plot(sizes[:len(avg_times)], avg_times, 'b-^', label='Average case', markersize=4)
            ax1.set_xlabel('Number of Words')
            ax1.set_ylabel('Insertion Time (seconds)')
            ax1.set_title('Insertion Performance')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xscale('log')
            ax1.set_yscale('log')
        
        # Plot 2: Search time vs size
        if self.results['search']['sizes']:
            ax2.plot(self.results['search']['sizes'], self.results['search']['avg_times'], 'b-o', markersize=4)
            ax2.set_xlabel('Number of Words')
            ax2.set_ylabel('Average Search Time (seconds)')
            ax2.set_title('Search Performance')
            ax2.grid(True, alpha=0.3)
            ax2.set_xscale('log')
            ax2.set_yscale('log')
        
        # Plot 3: Prefix search time vs size
        if self.results['prefix_search']['sizes']:
            ax3.plot(self.results['prefix_search']['sizes'], self.results['prefix_search']['avg_times'], 'g-o', markersize=4)
            ax3.set_xlabel('Number of Words')
            ax3.set_ylabel('Average Prefix Search Time (seconds)')
            ax3.set_title('Prefix Search Performance')
            ax3.grid(True, alpha=0.3)
            ax3.set_xscale('log')
            ax3.set_yscale('log')
        
        # Plot 4: Memory usage (node count) vs size
        if self.results['memory']['sizes']:
            ax4.plot(self.results['memory']['sizes'], self.results['memory']['node_counts'], 'r-o', markersize=4)
            ax4.set_xlabel('Number of Words')
            ax4.set_ylabel('Number of Nodes')
            ax4.set_title('Memory Usage (Node Count)')
            ax4.grid(True, alpha=0.3)
            ax4.set_xscale('log')
            ax4.set_yscale('log')
        
        plt.tight_layout()
        
        # Save plots
        plot_file = os.path.join(self.output_dir, "performance_plots.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Plots saved to {plot_file}")
        
        # Also save individual plots
        self.save_individual_plots()
    
    def save_individual_plots(self) -> None:
        """Save individual performance plots."""
        # Insertion performance comparison
        if self.results['insertion']['sizes']:
            plt.figure(figsize=(12, 8))
            sizes = self.results['insertion']['sizes'][::3]
            balanced_times = self.results['insertion']['times'][::3]
            worst_times = self.results['insertion']['times'][1::3]
            avg_times = self.results['insertion']['times'][2::3]
            
            plt.plot(sizes, balanced_times, 'g-o', label='Balanced case', linewidth=2, markersize=6)
            plt.plot(sizes[:len(worst_times)], worst_times, 'r-s', label='Worst case', linewidth=2, markersize=6)
            plt.plot(sizes[:len(avg_times)], avg_times, 'b-^', label='Average case', linewidth=2, markersize=6)
            
            plt.xlabel('Number of Words')
            plt.ylabel('Total Insertion Time (seconds)')
            plt.title('TST Insertion Performance: Best vs Average vs Worst Case (Up to 50K)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xscale('log')
            plt.yscale('log')
            
            plt.savefig(os.path.join(self.output_dir, "insertion_comparison.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # Complexity analysis plot
        if self.results['search']['sizes'] and self.results['insertion']['sizes']:
            plt.figure(figsize=(12, 8))
            
            # Theoretical complexity lines
            max_size = max(self.results['search']['sizes'])
            x_theory = np.logspace(1, np.log10(max_size), 100)
            
            # O(log n) line
            log_n = np.log2(x_theory)
            log_n_normalized = log_n / log_n[-1] * max(self.results['search']['avg_times'])
            
            # O(n) line  
            linear_n = x_theory
            linear_n_normalized = linear_n / linear_n[-1] * max(self.results['search']['avg_times']) * 10
            
            plt.subplot(2, 1, 1)
            plt.plot(self.results['search']['sizes'], self.results['search']['avg_times'], 
                    'bo-', label='Actual Search Time', linewidth=2, markersize=6)
            plt.plot(x_theory, log_n_normalized, 'g--', label='O(log n) theoretical', linewidth=2)
            plt.plot(x_theory, linear_n_normalized, 'r--', label='O(n) theoretical', linewidth=2)
            plt.xlabel('Number of Words')
            plt.ylabel('Average Search Time (seconds)')
            plt.title('Search Time Complexity Analysis (Up to 50K Words)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xscale('log')
            plt.yscale('log')
            
            plt.subplot(2, 1, 2)
            if len(self.results['insertion']['sizes']) >= 3:
                avg_insertion_times = self.results['insertion']['times'][2::3]
                avg_insertion_sizes = self.results['insertion']['sizes'][2::3]
                plt.plot(avg_insertion_sizes, avg_insertion_times, 'ro-', 
                        label='Actual Insertion Time', linewidth=2, markersize=6)
            
            plt.xlabel('Number of Words')
            plt.ylabel('Total Insertion Time (seconds)')
            plt.title('Insertion Time Complexity')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xscale('log')
            plt.yscale('log')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "complexity_analysis.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()


def compare_with_btree():
    """Compare TST performance with B-tree (simulated with sorted list)."""
    print("\nComparing with B-tree simulation...")
    
    sizes = [100, 500, 1000, 5000, 10000]
    tst_times = []
    btree_times = []
    
    for size in sizes:
        # Generate test data
        words = []
        for i in range(size):
            word = ''.join([chr(ord('a') + (i + j) % 26) for j in range(5)])
            words.append(word)
        
        # Test TST
        tst = TernarySearchTree()
        start_time = time.perf_counter()
        for word in words:
            tst.insert(word)
        # Search for some words
        search_count = min(1000, size // 10)
        for word in words[:search_count]:
            tst.search(word, exact=True)
        tst_time = time.perf_counter() - start_time
        tst_times.append(tst_time)
        
        # Test B-tree simulation (sorted list with binary search)
        btree_data = []
        start_time = time.perf_counter()
        for word in words:
            # Insert maintaining sorted order (simulates B-tree)
            import bisect
            bisect.insort(btree_data, word)
        # Search for some words
        for word in words[:search_count]:
            bisect.bisect_left(btree_data, word)
        btree_time = time.perf_counter() - start_time
        btree_times.append(btree_time)
        
        print(f"Size {size}: TST={tst_time:.4f}s, B-tree sim={btree_time:.4f}s")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, tst_times, 'b-o', label='Ternary Search Tree', linewidth=2, markersize=6)
    plt.plot(sizes, btree_times, 'r-s', label='B-tree (simulated)', linewidth=2, markersize=6)
    plt.xlabel('Number of Words')
    plt.ylabel('Total Time (seconds)')
    plt.title('TST vs B-tree Performance Comparison (Extended)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('tst_vs_btree_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main benchmarking function."""
    parser = argparse.ArgumentParser(description='Benchmark Ternary Search Tree performance')
    parser.add_argument('--output-dir', default='benchmark_results',
                       help='Directory to save results (default: benchmark_results)')
    parser.add_argument('--max-size', type=int, default=50000,
                       help='Maximum data size to test (default: 50000)')
    parser.add_argument('--compare-btree', action='store_true',
                       help='Include B-tree comparison')
    
    args = parser.parse_args()
    
    # Run benchmarks
    benchmark = TSTBenchmark(args.output_dir)
    benchmark.run_comprehensive_benchmark(args.max_size)
    
    # Compare with B-tree if requested
    if args.compare_btree:
        compare_with_btree()
    
    print(f"\nBenchmarking complete! Results saved in '{args.output_dir}'")


if __name__ == '__main__':
    main()