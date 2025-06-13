#!/bin/bash
#SBATCH --job-name=tst_benchmark_50k
#SBATCH --output=tst_benchmark_50k_%j.out
#SBATCH --error=tst_benchmark_50k_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB 
#SBATCH --clusters=genius
#SBATCH --account=lp_h_ds_students
#SBATCH --partition=batch

# TST Benchmarking Job Script for VSC HPC (50K Words)
echo "Starting TST benchmark job (50K words) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"

# Load required modules for VSC
module purge
module load cluster/genius/batch
module load matplotlib/3.8.2-gfbf-2023b

# Set up environment
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"

# Navigate to job directory
cd $SLURM_SUBMIT_DIR

# Verify environment
echo "Python version: $(python --version)"
echo "Loaded modules:"
module list

# Create output directory
OUTPUT_DIR="benchmark_results_50k_${SLURM_JOB_ID}"
mkdir -p $OUTPUT_DIR

echo "Working directory: $(pwd)"
echo "Output directory: $OUTPUT_DIR"

# Run the benchmark with 50K words
echo "Running 50K word benchmark..."
python benchmark_tst.py --output-dir "${OUTPUT_DIR}/results" --max-size 50000

# Run unit tests to ensure correctness
echo "Running unit tests..."
python -c "
import pytest
import sys
sys.exit(pytest.main(['-v', 'test_ternary_search_tree.py']))
" > "${OUTPUT_DIR}/test_results.txt" 2>&1

# Generate performance report
echo "Generating performance report..."
echo "50K word benchmark completed successfully" > "${OUTPUT_DIR}/summary.txt"
echo "Results saved in: $OUTPUT_DIR" >> "${OUTPUT_DIR}/summary.txt"
echo "Maximum dataset size: 50,000 words" >> "${OUTPUT_DIR}/summary.txt"
echo "Job completion time: $(date)" >> "${OUTPUT_DIR}/summary.txt"

echo "Benchmark completed at $(date)"
echo "Results in: $OUTPUT_DIR"

# List generated files
echo "Files generated:"
ls -la "${OUTPUT_DIR}"/