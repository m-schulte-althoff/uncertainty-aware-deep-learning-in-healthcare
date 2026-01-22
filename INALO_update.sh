#!/bin/bash
# Update INALO results by evaluating existing MIMIC-trained models on INALO test data
#
# This script:
#   1. Finds all trained models in ./output/in-hospital-mortality-fixed/
#   2. Evaluates each model on the INALO test dataset
#   3. Updates the combined_results.csv and combined_results.txt files
#
# Usage:
#   ./INALO_update.sh                    # Evaluate all models on INALO
#   ./INALO_update.sh -m lstm            # Evaluate only LSTM on INALO
#   ./INALO_update.sh -t probabilistic   # Evaluate only probabilistic models
#   ./INALO_update.sh -h                 # Show help

set -e  # Exit on error

# Get the directory where this script is located and cd to it
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default configuration
OUTPUT_DIR="./output"
INALO_DATA="./data/in-hospital-mortality-own-final/"
MIMIC_MODELS_DIR="./output/in-hospital-mortality-fixed"

# Default: evaluate all
SELECTED_MODELS=""
SELECTED_TYPES=""

# Help function
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Evaluate existing MIMIC-trained models on INALO test data and update combined results.

This script assumes models have already been trained on MIMIC data using run_all_experiments.sh.
It will:
  1. Find trained models in ./output/in-hospital-mortality-fixed/
  2. Evaluate each model on INALO test data
  3. Update combined_results.csv and combined_results.txt

Options:
  -m, --models MODELS       Comma-separated list of models to evaluate
                            Available: lstm, rnn, gru, transformer
                            Default: all models
  
  -t, --type TYPE           Model type(s) to evaluate
                            Available: deterministic, probabilistic, both
                            Default: both
  
  -o, --output DIR          Output directory (default: ./output)
  
  -h, --help                Show this help message

Examples:
  $0                                    # Evaluate all models on INALO
  $0 -m lstm                            # Evaluate only LSTM
  $0 -m lstm,gru -t probabilistic       # Evaluate LSTM and GRU probabilistic only
  $0 -t deterministic                   # Evaluate all deterministic models

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
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
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
    TYPES=("deterministic" "probabilistic")
elif [ "$SELECTED_TYPES" = "deterministic" ]; then
    TYPES=("deterministic")
elif [ "$SELECTED_TYPES" = "probabilistic" ]; then
    TYPES=("probabilistic")
else
    echo "Invalid type: $SELECTED_TYPES (use: deterministic, probabilistic, or both)"
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "./venv" ]; then
    echo "Activating virtual environment..."
    source ./venv/bin/activate
fi

# Disable wandb
export WANDB_MODE=disabled

echo "=============================================="
echo "INALO Evaluation Update"
echo "Started at $(date)"
echo "Working directory: $(pwd)"
echo "Output directory: $OUTPUT_DIR"
echo "Models: ${MODELS[*]}"
echo "Types: ${TYPES[*]}"
echo "INALO data: $INALO_DATA"
echo "=============================================="

# Function to evaluate a model on INALO
evaluate_on_inalo() {
    local model=$1
    local model_type=$2
    
    local model_dir="$MIMIC_MODELS_DIR/$model_type/$model"
    
    # Try to find a config file to determine which trial to use
    # First check for config_trial_*.json files (newer format)
    local config_file=$(ls "$model_dir"/config_trial_*.json 2>/dev/null | head -1)
    local trial_num=""
    
    if [ -n "$config_file" ]; then
        # Extract trial number from config filename
        trial_num=$(echo "$config_file" | grep -oP 'trial_\K\d+')
    fi
    
    # Find the corresponding model file
    local model_file=""
    if [ -n "$trial_num" ]; then
        model_file="$model_dir/final_model_trial_${trial_num}.pth"
    fi
    
    # Fallback: use first available model file
    if [ ! -f "$model_file" ]; then
        model_file=$(ls "$model_dir"/final_model_trial_*.pth 2>/dev/null | head -1)
    fi
    
    if [ -z "$model_file" ] || [ ! -f "$model_file" ]; then
        echo "Warning: No trained model found at $model_dir - skipping"
        return 1
    fi
    
    echo ""
    echo "----------------------------------------------"
    echo "Evaluating ${model_type^^} ${model^^} on INALO"
    echo "Model: $model_file"
    echo "----------------------------------------------"
    
    python ./ptsa/tasks/in_hospital_mortality/evaluate_model.py \
        --model_path "$model_file" \
        --model "$model" \
        --model_type "$model_type" \
        --eval_data "$INALO_DATA" \
        --eval_name "INALO" \
        --output_dir "$MIMIC_MODELS_DIR"
    
    echo "Completed: ${model^^} $model_type evaluation on INALO"
}

# Track start time
START_TIME=$(date +%s)

# Evaluate all selected models
for model in "${MODELS[@]}"; do
    for model_type in "${TYPES[@]}"; do
        evaluate_on_inalo "$model" "$model_type" || true
    done
done

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "=============================================="
echo "INALO evaluation completed at $(date)"
echo "Total runtime: ${MINUTES}m ${SECONDS}s"
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

models = ["lstm", "rnn", "gru", "transformer"]
types = ["deterministic", "probabilistic"]

# First, load existing combined_results.csv to preserve any existing data
existing_combined = Path(output_dir) / "combined_results.csv"
if existing_combined.exists():
    existing_df = pd.read_csv(existing_combined)
    results.append(existing_df)
    print(f"Loaded {len(existing_df)} existing rows from combined_results.csv")

# Collect MIMIC results (from training) and INALO results (from evaluation)
for model_type in types:
    for model in models:
        # MIMIC results from training
        mimic_csv = Path(output_dir) / "in-hospital-mortality-fixed" / model_type / model / "results_table.csv"
        if mimic_csv.exists():
            df = pd.read_csv(mimic_csv)
            results.append(df)
        else:
            print(f"Warning: Missing MIMIC results at {mimic_csv}")
        
        # INALO results from evaluation
        inalo_csv = Path(output_dir) / "in-hospital-mortality-fixed" / model_type / model / "eval_inalo" / "results_table.csv"
        if inalo_csv.exists():
            df = pd.read_csv(inalo_csv)
            results.append(df)
        else:
            print(f"Warning: Missing INALO evaluation results at {inalo_csv}")

if results:
    # Combine all results
    combined_df = pd.concat(results, ignore_index=True)
    
    # Remove duplicates (keep latest)
    combined_df = combined_df.drop_duplicates(subset=['Model', 'Dataset'], keep='last')
    
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
        f.write("COMBINED RESULTS TABLE\n")
        f.write("Note: All models were trained on MIMIC and evaluated on both MIMIC and INALO\n")
        f.write("=" * 80 + "\n\n")
        f.write(combined_df.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("COMBINED RESULTS TABLE (Updated)")
    print("=" * 80)
    print("Note: All models were trained on MIMIC and evaluated on both MIMIC and INALO")
    print("")
    print(combined_df.to_string(index=False))
    print("\n")
    print(f"Results saved to: {combined_path}")
    print(f"Results saved to: {txt_path}")
else:
    print("No results found to aggregate!")
EOF

echo ""
echo "Done! Check $OUTPUT_DIR/combined_results.csv for the updated table."
