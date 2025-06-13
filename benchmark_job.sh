#!/bin/bash
#SBATCH --job-name=tst_benchmark
#SBATCH --output=tst_benchmark_%j.out
#SBATCH --error=tst_benchmark_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --partition=batch
#SBATCH --gpus-per-node=1

# TST Benchmarking Job Script for HPC
# This script runs comprehensive performance benchmarks for the Ternary Search Tree implementation

echo "Starting TST benchmark job at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"

# Load required modules (adjust according to your HPC environment)
module purge

# Set up environment
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"

# Navigate to job directory
cd $SLURM_SUBMIT_DIR

# Create output directory
OUTPUT_DIR="benchmark_results_${SLURM_JOB_ID}"
mkdir -p $OUTPUT_DIR

echo "Working directory: $(pwd)"
echo "Output directory: $OUTPUT_DIR"

# Run the benchmark with different configurations
echo "Running basic benchmark..."
python benchmark_tst.py --output-dir "${OUTPUT_DIR}/basic" --max-size 50000

echo "Running extended benchmark with B-tree comparison..."
python benchmark_tst.py --output-dir "${OUTPUT_DIR}/extended" --max-size 100000 --compare-btree

# Run unit tests to ensure correctness
echo "Running unit tests..."
python -m pytest test_ternary_search_tree.py -v > "${OUTPUT_DIR}/test_results.txt" 2>&1

# Generate performance report
echo "Generating performance report..."
python -c "
import json
import os

def generate_report(results_dir):
    results_file = os.path.join(results_dir, 'benchmark_results.json')
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        report_file = os.path.join(results_dir, 'performance_report.txt')
        with open(report_file, 'w') as f:
            f.write('Ternary Search Tree Performance Report\n')
            f.write('=' * 40 + '\n\n')
            
            # Insertion performance
            if results['insertion']['sizes']:
                f.write('Insertion Performance:\n')
                for i in range(0, len(results['insertion']['sizes']), 3):
                    size = results['insertion']['sizes'][i]
                    balanced_time = results['insertion']['times'][i]
                    worst_time = results['insertion']['times'][i+1] if i+1 < len(results['insertion']['times']) else 0
                    avg_time = results['insertion']['times'][i+2] if i+2 < len(results['insertion']['times']) else 0
                    
                    f.write(f'  Size {size:5d}: Balanced={balanced_time:.4f}s, Worst={worst_time:.4f}s, Average={avg_time:.4f}s\n')
                f.write('\n')
            
            # Search performance
            if results['search']['sizes']:
                f.write('Search Performance:\n')
                for size, avg_time in zip(results['search']['sizes'], results['search']['avg_times']):
                    f.write(f'  Size {size:5d}: {avg_time*1000:.4f}ms average\n')
                f.write('\n')
            
            # Memory usage
            if results['memory']['sizes']:
                f.write('Memory Usage (Node Count):\n')
                for size, nodes in zip(results['memory']['sizes'], results['memory']['node_counts']):
                    ratio = nodes / size if size > 0 else 0
                    f.write(f'  Size {size:5d}: {nodes:6d} nodes ({ratio:.2f} nodes/word)\n')

# Generate reports for both benchmark runs
generate_report('${OUTPUT_DIR}/basic')
generate_report('${OUTPUT_DIR}/extended')
"

# Collect system information
echo "Collecting system information..."
cat > "${OUTPUT_DIR}/system_info.txt" << EOF
System Information
==================
Date: $(date)
Node: $SLURM_NODELIST
Job ID: $SLURM_JOB_ID
CPUs: $SLURM_CPUS_PER_TASK
Memory: $SLURM_MEM_PER_NODE MB

Python Version:
$(python --version)

CPU Info:
$(lscpu | grep "Model name" || echo "CPU info not available")

Memory Info:
$(free -h)

Loaded Modules:
$(module list 2>&1)
EOF

# Create archive of all results
echo "Creating results archive..."
tar -czf "tst_benchmark_results_${SLURM_JOB_ID}.tar.gz" "$OUTPUT_DIR"

echo "Benchmark completed at $(date)"
echo "Results archived in: tst_benchmark_results_${SLURM_JOB_ID}.tar.gz"

# Print summary
echo ""
echo "=== Benchmark Summary ==="
if [ -f "${OUTPUT_DIR}/basic/performance_report.txt" ]; then
    echo "Basic benchmark report:"
    head -20 "${OUTPUT_DIR}/basic/performance_report.txt"
fi

echo ""
echo "Files generated:"
ls -la "${OUTPUT_DIR}"/*