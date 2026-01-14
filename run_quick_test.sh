#!/bin/bash
# Quick test script - runs with limited data samples for sanity checking
# Use this to verify everything works before running the full experiments
#
# Usage: ./run_quick_test.sh
# Make sure to run from the project root directory

set -e  # Exit on error

# Get the directory where this script is located and cd to it
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration (relative paths)
OUTPUT_DIR="./output_test"
NUM_TRIALS=1  # Single trial for quick testing
MIMIC_DATA="./data/in-hospital-mortality-fixed/"
INALO_DATA="./data/in-hospital-mortality-own-final/"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Activate virtual environment if it exists
if [ -d "./venv" ]; then
    echo "Activating virtual environment..."
    source ./venv/bin/activate
fi

# Disable wandb
export WANDB_MODE=disabled

echo "=============================================="
echo "Running quick sanity check with --small_part flag"
echo "Working directory: $(pwd)"
echo "=============================================="

# Test one model on each dataset (deterministic + probabilistic)
# Using LSTM as test case

echo ""
echo "Testing LSTM deterministic on MIMIC..."
python ./ptsa/tasks/in_hospital_mortality/train_deterministic.py \
    --network "./ptsa/models/deterministic/lstm_classification.py" \
    --partition custom \
    --data "$MIMIC_DATA" \
    --model lstm \
    --model_name lstm_test.pth \
    --output_dir "$OUTPUT_DIR" \
    --num_trials $NUM_TRIALS \
    --small_part

echo ""
echo "Testing LSTM probabilistic on MIMIC..."
python ./ptsa/tasks/in_hospital_mortality/train_probabilistic.py \
    --network "./ptsa/models/probabilistic/lstm_classification.py" \
    --partition custom \
    --data "$MIMIC_DATA" \
    --model lstm \
    --model_name lstm_prob_test.pth \
    --output_dir "$OUTPUT_DIR" \
    --num_trials $NUM_TRIALS \
    --small_part

echo ""
echo "Testing LSTM deterministic on INALO..."
python ./ptsa/tasks/in_hospital_mortality/train_deterministic.py \
    --network "./ptsa/models/deterministic/lstm_classification.py" \
    --partition custom \
    --data "$INALO_DATA" \
    --model lstm \
    --model_name lstm_test.pth \
    --output_dir "$OUTPUT_DIR" \
    --num_trials $NUM_TRIALS \
    --small_part

echo ""
echo "Testing LSTM probabilistic on INALO..."
python ./ptsa/tasks/in_hospital_mortality/train_probabilistic.py \
    --network "./ptsa/models/probabilistic/lstm_classification.py" \
    --partition custom \
    --data "$INALO_DATA" \
    --model lstm \
    --model_name lstm_prob_test.pth \
    --output_dir "$OUTPUT_DIR" \
    --num_trials $NUM_TRIALS \
    --small_part

echo ""
echo "=============================================="
echo "Quick test completed!"
echo "Check output in: $OUTPUT_DIR"
echo "=============================================="

# Show results
echo ""
echo "Results:"
find "$OUTPUT_DIR" -name "results_table.txt" -exec echo "--- {} ---" \; -exec cat {} \; -exec echo "" \;
