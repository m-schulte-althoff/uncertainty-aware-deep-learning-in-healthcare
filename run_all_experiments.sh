#!/bin/bash
# Run all experiments to fill out the results table
# This script trains all model/dataset combinations for both deterministic and probabilistic versions
#
# Usage: 
#   ./run_all_experiments.sh                     # Run all experiments
#   ./run_all_experiments.sh -m lstm             # Run only LSTM model
#   ./run_all_experiments.sh -m lstm,gru         # Run LSTM and GRU models
#   ./run_all_experiments.sh -t deterministic    # Run only deterministic training
#   ./run_all_experiments.sh -t probabilistic    # Run only probabilistic training
#   ./run_all_experiments.sh -d MIMIC            # Run only on MIMIC dataset
#   ./run_all_experiments.sh -d INALO            # Run only on INALO dataset
#   ./run_all_experiments.sh -m lstm -t probabilistic -d MIMIC  # Combined filters
#   ./run_all_experiments.sh -n 5                # Set number of Optuna trials
#   ./run_all_experiments.sh --small             # Use --small_part flag for quick testing
#   ./run_all_experiments.sh -h                  # Show help

set -e  # Exit on error

# Get the directory where this script is located and cd to it
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default configuration
OUTPUT_DIR="./output"
NUM_TRIALS=10
MIMIC_DATA="./data/in-hospital-mortality-fixed/"
INALO_DATA="./data/in-hospital-mortality-own-final/"
SMALL_PART=""

# Default: run all
SELECTED_MODELS=""
SELECTED_TYPES=""
SELECTED_DATASETS=""

# Help function
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Run training experiments for in-hospital mortality prediction.

Options:
  -m, --models MODELS       Comma-separated list of models to train
                            Available: lstm, rnn, gru, transformer
                            Default: all models
  
  -t, --type TYPE           Training type to run
                            Available: deterministic, probabilistic, both
                            Default: both
  
  -d, --datasets DATASETS   Comma-separated list of datasets to use
                            Available: MIMIC, INALO
                            Default: both datasets
  
  -n, --num-trials N        Number of Optuna trials (default: 10)
  
  -o, --output DIR          Output directory (default: ./output)
  
  --small                   Use --small_part flag for quick testing with limited data
  
  -h, --help                Show this help message

Examples:
  $0                                    # Run all experiments
  $0 -m lstm                            # Run only LSTM
  $0 -m lstm,gru -t probabilistic       # Run LSTM and GRU probabilistic only
  $0 -d MIMIC -t deterministic          # Run all deterministic models on MIMIC
  $0 -m transformer -d INALO --small    # Quick test transformer on INALO
  $0 -n 5                               # Run all with 5 trials instead of 10

EOF
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--models)
            SELECTED_MODELS="$2"
            shift 2
            ;;
        -t|--type)
            SELECTED_TYPES="$2"
            shift 2
            ;;
        -d|--datasets)
            SELECTED_DATASETS="$2"
            shift 2
            ;;
        -n|--num-trials)
            NUM_TRIALS="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --small)
            SMALL_PART="--small_part"
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Set defaults if not specified
if [ -z "$SELECTED_MODELS" ]; then
    MODELS=("lstm" "rnn" "gru" "transformer")
else
    IFS=',' read -ra MODELS <<< "$SELECTED_MODELS"
fi

if [ -z "$SELECTED_TYPES" ] || [ "$SELECTED_TYPES" = "both" ]; then
    RUN_DETERMINISTIC=true
    RUN_PROBABILISTIC=true
elif [ "$SELECTED_TYPES" = "deterministic" ]; then
    RUN_DETERMINISTIC=true
    RUN_PROBABILISTIC=false
elif [ "$SELECTED_TYPES" = "probabilistic" ]; then
    RUN_DETERMINISTIC=false
    RUN_PROBABILISTIC=true
else
    echo "Invalid type: $SELECTED_TYPES (use: deterministic, probabilistic, or both)"
    exit 1
fi

if [ -z "$SELECTED_DATASETS" ]; then
    RUN_MIMIC=true
    RUN_INALO=true
else
    RUN_MIMIC=false
    RUN_INALO=false
    IFS=',' read -ra DATASET_LIST <<< "$SELECTED_DATASETS"
    for ds in "${DATASET_LIST[@]}"; do
        case "$ds" in
            MIMIC) RUN_MIMIC=true ;;
            INALO) RUN_INALO=true ;;
            *) echo "Invalid dataset: $ds (use: MIMIC, INALO)"; exit 1 ;;
        esac
    done
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Activate virtual environment if it exists
if [ -d "./venv" ]; then
    echo "Activating virtual environment..."
    source ./venv/bin/activate
fi

# Disable wandb if not logged in (comment out if you want wandb logging)
export WANDB_MODE=disabled

echo "=============================================="
echo "Starting experiments at $(date)"
echo "Working directory: $(pwd)"
echo "Output directory: $OUTPUT_DIR"
echo "Number of Optuna trials: $NUM_TRIALS"
echo "Models: ${MODELS[*]}"
echo "Deterministic: $RUN_DETERMINISTIC"
echo "Probabilistic: $RUN_PROBABILISTIC"
echo "MIMIC dataset: $RUN_MIMIC"
echo "INALO dataset: $RUN_INALO"
[ -n "$SMALL_PART" ] && echo "Using --small_part for quick testing"
echo "=============================================="

# Function to run deterministic training
run_deterministic() {
    local model=$1
    local dataset_name=$2
    local data_path=$3
    
    echo ""
    echo "----------------------------------------------"
    echo "Training DETERMINISTIC ${model^^} on $dataset_name"
    echo "----------------------------------------------"
    
    python ./ptsa/tasks/in_hospital_mortality/train_deterministic.py \
        --network "./ptsa/models/deterministic/${model}_classification.py" \
        --partition custom \
        --data "$data_path" \
        --model "$model" \
        --model_name "${model}_deterministic.pth" \
        --output_dir "$OUTPUT_DIR" \
        --num_trials "$NUM_TRIALS" \
        $SMALL_PART
    
    echo "Completed: ${model^^} deterministic on $dataset_name"
}

# Function to run probabilistic training
run_probabilistic() {
    local model=$1
    local dataset_name=$2
    local data_path=$3
    
    echo ""
    echo "----------------------------------------------"
    echo "Training PROBABILISTIC ${model^^} + MC Dropout on $dataset_name"
    echo "----------------------------------------------"
    
    python ./ptsa/tasks/in_hospital_mortality/train_probabilistic.py \
        --network "./ptsa/models/probabilistic/${model}_classification.py" \
        --partition custom \
        --data "$data_path" \
        --model "$model" \
        --model_name "${model}_probabilistic.pth" \
        --output_dir "$OUTPUT_DIR" \
        --num_trials "$NUM_TRIALS" \
        $SMALL_PART
    
    echo "Completed: ${model^^} + MC Dropout probabilistic on $dataset_name"
}

# Track start time
START_TIME=$(date +%s)

# Run experiments based on selected options
for model in "${MODELS[@]}"; do
    # Deterministic models (no uncertainty)
    if [ "$RUN_DETERMINISTIC" = true ]; then
        if [ "$RUN_MIMIC" = true ]; then
            run_deterministic "$model" "MIMIC" "$MIMIC_DATA"
        fi
        if [ "$RUN_INALO" = true ]; then
            run_deterministic "$model" "INALO" "$INALO_DATA"
        fi
    fi
    
    # Probabilistic models (with MC Dropout uncertainty)
    if [ "$RUN_PROBABILISTIC" = true ]; then
        if [ "$RUN_MIMIC" = true ]; then
            run_probabilistic "$model" "MIMIC" "$MIMIC_DATA"
        fi
        if [ "$RUN_INALO" = true ]; then
            run_probabilistic "$model" "INALO" "$INALO_DATA"
        fi
    fi
done

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "=============================================="
echo "All experiments completed at $(date)"
echo "Total runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "=============================================="

# Aggregate all results into a single table
echo ""
echo "Aggregating results..."

python << 'EOF'
import os
import pandas as pd
from pathlib import Path

output_dir = "./output"
results = []

# Define expected structure
models = ["lstm", "rnn", "gru", "transformer"]
datasets = {
    "in-hospital-mortality-fixed": "MIMIC",
    "in-hospital-mortality-own-final": "INALO"
}
types = ["deterministic", "probabilistic"]

for dataset_dir, dataset_display in datasets.items():
    for model_type in types:
        for model in models:
            csv_path = Path(output_dir) / dataset_dir / model_type / model / "results_table.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                results.append(df)
            else:
                print(f"Warning: Missing results at {csv_path}")

if results:
    # Combine all results
    combined_df = pd.concat(results, ignore_index=True)
    
    # Sort by Model and Dataset for proper table order
    model_order = ["LSTM", "LSTM + MC Dropout", "RNN", "RNN + MC Dropout", 
                   "GRU", "GRU + MC Dropout", "TRANSFORMER", "TRANSFORMER + MC Dropout"]
    combined_df["Model_Order"] = combined_df["Model"].apply(
        lambda x: model_order.index(x) if x in model_order else 999
    )
    combined_df = combined_df.sort_values(["Model_Order", "Dataset"]).drop("Model_Order", axis=1)
    
    # Save combined results
    combined_path = Path(output_dir) / "combined_results.csv"
    combined_df.to_csv(combined_path, index=False)
    
    txt_path = Path(output_dir) / "combined_results.txt"
    with open(txt_path, 'w') as f:
        f.write(combined_df.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("COMBINED RESULTS TABLE")
    print("=" * 80)
    print(combined_df.to_string(index=False))
    print("\n")
    print(f"Results saved to: {combined_path}")
    print(f"Results saved to: {txt_path}")
else:
    print("No results found to aggregate!")
EOF

echo ""
echo "Done! Check $OUTPUT_DIR/combined_results.csv for the full table."
