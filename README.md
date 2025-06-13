# Ternary Search Tree Implementation

A Python implementation of a Ternary Search Tree (TST) data structure for efficient string storage, retrieval, and prefix matching operations.

## Repository Contents

### Core Implementation
- `ternary_search_tree/` - Main package directory
  - `__init__.py` - Package initialization and exports
  - `node.py` - TSTNode class for individual tree nodes
  - `tree.py` - TernarySearchTree main implementation
  - `__pycache__/` - Compiled Python bytecode (auto-generated)

### Testing and Validation
- `test_ternary_search_tree.py` - Comprehensive unit tests
- `ternary_search_tree.ipynb` - Jupyter notebook with examples and demonstrations

### Benchmarking and Performance
- `benchmark_tst.py` - Performance benchmarking script with plotting
- `benchmark_job.sh` - SLURM job script for HPC cluster execution
- `benchmark_results_*/` - Generated benchmark results (example from job 58226647)
  - `basic/` - Benchmark data and plots
    - `benchmark_results.json` - Raw performance data
    - `performance_plots.png` - Combined performance visualization
    - `insertion_comparison.png` - Insert performance comparison
    - `complexity_analysis.png` - Time complexity analysis
  - `test_results.txt` - Unit test execution results
  - `summary.txt` - Benchmark summary
  - `tst_benchmark_*.out/.err` - Job execution logs

### Configuration
- `requirements.txt` - Python package dependencies
- `README.md` - This documentation file

### Analysis of Results
- `Analysis.md` - Benchmarks analysis and conclusions

## Features

The Ternary Search Tree implementation provides:

- **String Operations**: Insert, search (exact and prefix), delete
- **Memory Efficient**: More space-efficient than standard tries
- **Flexible Search**: Support for both exact matches and prefix searches
- **Comprehensive Testing**: Full unit test suite with edge cases
- **Performance Analysis**: Benchmarking tools with visualization
- **HPC Compatible**: Ready for high-performance computing environments