#!/bin/bash
# Run all experiments to fill out the results table
# This script trains models on MIMIC data and evaluates on both MIMIC and INALO test sets
#
# IMPORTANT: Models are ONLY trained on MIMIC data. They are then evaluated on:
#   1. MIMIC test set (in-domain evaluation)
#   2. INALO test set (out-of-domain/transfer evaluation)
#
# Usage: 
#   ./run_all_experiments.sh                     # Run all experiments
#   ./run_all_experiments.sh -m lstm             # Run only LSTM model
#   ./run_all_experiments.sh -m lstm,gru         # Run LSTM and GRU models
#   ./run_all_experiments.sh -t deterministic    # Run only deterministic training
#   ./run_all_experiments.sh -t probabilistic    # Run only probabilistic training
#   ./run_all_experiments.sh --eval-only         # Skip training, only run evaluation
#   ./run_all_experiments.sh -m lstm -t probabilistic  # Combined filters
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
EVAL_ONLY=false

# Default: run all
SELECTED_MODELS=""
SELECTED_TYPES=""

# Help function
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Run training experiments for in-hospital mortality prediction.

IMPORTANT: Models are ONLY trained on MIMIC data. They are then evaluated on:
  1. MIMIC test set (in-domain evaluation)  
  2. INALO test set (out-of-domain/transfer evaluation)

Options:
  -m, --models MODELS       Comma-separated list of models to train
                            Available: lstm, rnn, gru, transformer
                            Default: all models
  
  -t, --type TYPE           Training type to run
                            Available: deterministic, probabilistic, both
                            Default: both
  
  -n, --num-trials N        Number of Optuna trials (default: 10)
  
  -o, --output DIR          Output directory (default: ./output)
  
  --small                   Use --small_part flag for quick testing with limited data
  
  --eval-only               Skip training, only run evaluation on INALO
                            (assumes models are already trained)
  
  -h, --help                Show this help message

Examples:
  $0                                    # Train all models on MIMIC, eval on MIMIC+INALO
  $0 -m lstm                            # Train only LSTM on MIMIC, eval on MIMIC+INALO
  $0 -m lstm,gru -t probabilistic       # Train LSTM and GRU probabilistic only
  $0 --eval-only                        # Only evaluate existing models on INALO
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
        --eval-only)
            EVAL_ONLY=true
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
echo "Training on: MIMIC (always)"
echo "Evaluating on: MIMIC + INALO"
[ "$EVAL_ONLY" = true ] && echo "Mode: EVALUATION ONLY (skipping training)"
[ -n "$SMALL_PART" ] && echo "Using --small_part for quick testing"
echo "=============================================="

# Function to run deterministic training (always on MIMIC)
run_deterministic() {
    local model=$1
    
    echo ""
    echo "----------------------------------------------"
    echo "Training DETERMINISTIC ${model^^} on MIMIC"
    echo "----------------------------------------------"
    
    python ./ptsa/tasks/in_hospital_mortality/train_deterministic.py \
        --network "./ptsa/models/deterministic/${model}_classification.py" \
        --partition custom \
        --data "$MIMIC_DATA" \
        --model "$model" \
        --model_name "${model}_deterministic.pth" \
        --output_dir "$OUTPUT_DIR" \
        --num_trials "$NUM_TRIALS" \
        $SMALL_PART
    
    echo "Completed: ${model^^} deterministic training on MIMIC"
}

# Function to run probabilistic training (always on MIMIC)
run_probabilistic() {
    local model=$1
    
    echo ""
    echo "----------------------------------------------"
    echo "Training PROBABILISTIC ${model^^} + MC Dropout on MIMIC"
    echo "----------------------------------------------"
    
    python ./ptsa/tasks/in_hospital_mortality/train_probabilistic.py \
        --network "./ptsa/models/probabilistic/${model}_classification.py" \
        --partition custom \
        --data "$MIMIC_DATA" \
        --model "$model" \
        --model_name "${model}_probabilistic.pth" \
        --output_dir "$OUTPUT_DIR" \
        --num_trials "$NUM_TRIALS" \
        $SMALL_PART
    
    echo "Completed: ${model^^} + MC Dropout probabilistic training on MIMIC"
}

# Function to evaluate a trained model on INALO
evaluate_on_inalo() {
    local model=$1
    local model_type=$2  # deterministic or probabilistic
    
    # Find the best model file
    local model_dir="$OUTPUT_DIR/in-hospital-mortality-fixed/$model_type/$model"
    
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
        echo "Warning: No trained model found at $model_dir"
        return 1
    fi
    
    echo ""
    echo "----------------------------------------------"
    echo "Evaluating ${model_type^^} ${model^^} on INALO"
    echo "Using model: $model_file"
    echo "----------------------------------------------"
    
    python ./ptsa/tasks/in_hospital_mortality/evaluate_model.py \
        --model_path "$model_file" \
        --model "$model" \
        --model_type "$model_type" \
        --eval_data "$INALO_DATA" \
        --eval_name "INALO" \
        --output_dir "$OUTPUT_DIR/in-hospital-mortality-fixed"
    
    echo "Completed: ${model^^} $model_type evaluation on INALO"
}

# Function to evaluate a trained model on MIMIC test set (for calibration curves)
evaluate_on_mimic() {
    local model=$1
    local model_type=$2  # deterministic or probabilistic
    
    # Find the best model file
    local model_dir="$OUTPUT_DIR/in-hospital-mortality-fixed/$model_type/$model"
    
    local config_file=$(ls "$model_dir"/config_trial_*.json 2>/dev/null | head -1)
    local trial_num=""
    
    if [ -n "$config_file" ]; then
        trial_num=$(echo "$config_file" | grep -oP 'trial_\K\d+')
    fi
    
    local model_file=""
    if [ -n "$trial_num" ]; then
        model_file="$model_dir/final_model_trial_${trial_num}.pth"
    fi
    
    if [ ! -f "$model_file" ]; then
        model_file=$(ls "$model_dir"/final_model_trial_*.pth 2>/dev/null | head -1)
    fi
    
    if [ -z "$model_file" ] || [ ! -f "$model_file" ]; then
        echo "Warning: No trained model found at $model_dir - skipping MIMIC eval"
        return 1
    fi
    
    echo ""
    echo "----------------------------------------------"
    echo "Evaluating ${model_type^^} ${model^^} on MIMIC (calibration)"
    echo "Using model: $model_file"
    echo "----------------------------------------------"
    
    python ./ptsa/tasks/in_hospital_mortality/evaluate_model.py \
        --model_path "$model_file" \
        --model "$model" \
        --model_type "$model_type" \
        --eval_data "$MIMIC_DATA" \
        --eval_name "MIMIC" \
        --output_dir "$OUTPUT_DIR/in-hospital-mortality-fixed"
    
    echo "Completed: ${model^^} $model_type evaluation on MIMIC"
}

# Track start time
START_TIME=$(date +%s)

# Run experiments based on selected options
for model in "${MODELS[@]}"; do
    # Deterministic models (no uncertainty)
    if [ "$RUN_DETERMINISTIC" = true ]; then
        # Train on MIMIC (unless --eval-only)
        if [ "$EVAL_ONLY" = false ]; then
            run_deterministic "$model"
        fi
        
        # Evaluate on MIMIC test set (calibration curves)
        evaluate_on_mimic "$model" "deterministic"
        # Evaluate on INALO
        evaluate_on_inalo "$model" "deterministic"
    fi
    
    # Probabilistic models (with MC Dropout uncertainty)
    if [ "$RUN_PROBABILISTIC" = true ]; then
        # Train on MIMIC (unless --eval-only)
        if [ "$EVAL_ONLY" = false ]; then
            run_probabilistic "$model"
        fi
        
        # Evaluate on MIMIC test set (calibration curves)
        evaluate_on_mimic "$model" "probabilistic"
        # Evaluate on INALO
        evaluate_on_inalo "$model" "probabilistic"
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

# Models trained on MIMIC are stored in: output/in-hospital-mortality-fixed/{type}/{model}/
# INALO evaluations are stored in: output/in-hospital-mortality-fixed/{type}/{model}/eval_inalo/

models = ["lstm", "rnn", "gru", "transformer"]
types = ["deterministic", "probabilistic"]

# Collect MIMIC results (from training)
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
    print("Note: All models were trained on MIMIC and evaluated on both MIMIC and INALO")
    print("")
    print(combined_df.to_string(index=False))
    print("\n")
    print(f"Results saved to: {combined_path}")
    print(f"Results saved to: {txt_path}")
else:
    print("No results found to aggregate!")
EOF

# Generate combined calibration curves overview
echo ""
echo "Generating combined calibration curve overview..."

python << 'CALEOF'
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

output_dir = "./output"
base = Path(output_dir) / "in-hospital-mortality-fixed"

models = ["lstm", "rnn", "gru", "transformer"]
types = ["deterministic", "probabilistic"]
datasets = ["mimic", "inalo"]

# Collect all calibration curve images
cal_files = []
for model_type in types:
    for model in models:
        for ds in datasets:
            cal_path = base / model_type / model / f"eval_{ds}" / "calibration_curve.png"
            if cal_path.exists():
                if model_type == "probabilistic":
                    label = f"{model.upper()} + MC Dropout"
                else:
                    label = model.upper()
                cal_files.append({
                    'path': str(cal_path),
                    'model': label,
                    'dataset': ds.upper(),
                    'type': model_type
                })

if cal_files:
    n = len(cal_files)
    # Arrange in a grid: rows = models, cols = datasets
    ncols = 2  # MIMIC, INALO
    nrows = (n + ncols - 1) // ncols
    nrows = max(nrows, 1)

    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 8 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    # Track which subplots are used
    used = set()
    row = 0
    for model_type in types:
        for model in models:
            for col, ds in enumerate(datasets):
                cal_path = base / model_type / model / f"eval_{ds}" / "calibration_curve.png"
                if cal_path.exists():
                    img = plt.imread(str(cal_path))
                    if row < nrows:
                        axes[row, col].imshow(img)
                        axes[row, col].axis('off')
                        used.add((row, col))
            if any((base / model_type / model / f"eval_{ds}" / "calibration_curve.png").exists() for ds in datasets):
                row += 1

    # Hide unused axes
    for r in range(nrows):
        for c in range(ncols):
            if (r, c) not in used:
                axes[r, c].axis('off')

    plt.suptitle('Calibration Curves — All Models', fontsize=18, y=1.01)
    plt.tight_layout()
    overview_path = Path(output_dir) / "calibration_curves_overview.png"
    fig.savefig(str(overview_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Combined calibration overview saved to: {overview_path}")
    print(f"Individual calibration curves are in each model's eval_*/calibration_curve.png")
else:
    print("No calibration curve images found to combine.")
CALEOF

echo ""
echo "Done! Check $OUTPUT_DIR/combined_results.csv for the full table."
echo "Check $OUTPUT_DIR/calibration_curves_overview.png for calibration curves."
