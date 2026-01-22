#!/usr/bin/env python
"""
Evaluate a trained model on a specified test dataset.

This script loads a pre-trained model and evaluates it on a test dataset,
saving metrics to the appropriate output directory.

Usage:
    python evaluate_model.py \
        --model_path ./output/in-hospital-mortality-fixed/probabilistic/lstm/final_model_trial_0.pth \
        --model lstm \
        --model_type probabilistic \
        --eval_data ./data/in-hospital-mortality-own-final \
        --eval_name INALO \
        --output_dir ./output
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    roc_auc_score, f1_score, average_precision_score
)

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from ptsa.tasks.readers import InHospitalMortalityReader
from ptsa.utils.preprocessing import Discretizer, Normalizer
from ptsa.tasks.in_hospital_mortality.utils import load_data

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def remove_columns(data, header, columns_to_remove):
    """Remove specified columns from the data."""
    header_list = header if isinstance(header, list) else header.split(',')
    indices_to_keep = [i for i, col in enumerate(header_list) 
                       if not any(remove_col in col for remove_col in columns_to_remove)]
    
    new_data = []
    for sample in data[0]:
        new_sample = sample[:, indices_to_keep]
        new_data.append(new_sample)
    
    return (new_data, data[1])


def load_model(model_name, model_type, model_path, config, device):
    """Load a trained model."""
    if model_type == "deterministic":
        if model_name == "lstm":
            from ptsa.models.deterministic.lstm_classification import LSTM
            model = LSTM(config["input_size"], config["hidden_size"], 
                        config["num_layers"], config["dropout"]).to(device)
        elif model_name == "rnn":
            from ptsa.models.deterministic.rnn_classification import RNN
            model = RNN(config["input_size"], config["hidden_size"], 
                       config["num_layers"], config["dropout"]).to(device)
        elif model_name == "gru":
            from ptsa.models.deterministic.gru_classification import GRU
            model = GRU(config["input_size"], config["hidden_size"], 
                       config["num_layers"], config["dropout"]).to(device)
        elif model_name == "transformer":
            from ptsa.models.deterministic.transformer_classification import TransformerIHM
            model = TransformerIHM(
                input_size=config["input_size"],
                d_model=config["d_model"],
                nhead=config["nhead"],
                num_layers=config["num_layers"],
                dropout=config["dropout"],
                dim_feedforward=config["dim_feedforward"]
            ).to(device)
    else:  # probabilistic
        if model_name == "lstm":
            from ptsa.models.probabilistic.lstm_classification import LSTM
            model = LSTM(config["input_size"], config["hidden_size"], 
                        config["num_layers"], config["dropout"]).to(device)
        elif model_name == "rnn":
            from ptsa.models.probabilistic.rnn_classification import RNN
            model = RNN(config["input_size"], config["hidden_size"], 
                       config["num_layers"], config["dropout"]).to(device)
        elif model_name == "gru":
            from ptsa.models.probabilistic.gru_classification import GRU
            model = GRU(config["input_size"], config["hidden_size"], 
                       config["num_layers"], config["dropout"]).to(device)
        elif model_name == "transformer":
            from ptsa.models.probabilistic.transformer_classification import TransformerIHM
            model = TransformerIHM(
                input_size=config["input_size"],
                d_model=config["d_model"],
                nhead=config["nhead"],
                num_layers=config["num_layers"],
                dropout=config["dropout"],
                dim_feedforward=config["dim_feedforward"]
            ).to(device)
    
    model.load_state_dict(torch.load(model_path, weights_only=True))
    return model


def load_test_data(data_path, timestep=1.0):
    """Load and preprocess test data from specified path."""
    test_reader = InHospitalMortalityReader(
        dataset_dir=os.path.join(data_path, 'test'), 
        listfile=os.path.join(data_path, 'test/listfile.csv'), 
        period_length=48.0
    )
    
    # We need a sample from train to get discretizer header
    train_reader = InHospitalMortalityReader(
        dataset_dir=os.path.join(data_path, 'train'), 
        listfile=os.path.join(data_path, 'train/listfile.csv'), 
        period_length=48.0
    )
    
    discretizer = Discretizer(
        timestep=float(timestep), 
        store_masks=True, 
        impute_strategy='previous',
        start_time='zero'
    )
    discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
    cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

    # Load normalizer (using the pre-computed one for now)
    normalizer = Normalizer(fields=cont_channels)
    normalizer_state = f'ihm_ts{timestep}.input_str_previous.start_time_zero.normalizer'
    normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
    normalizer.load_params(normalizer_state)
    
    columns_to_remove = [
        "Glascow coma scale motor response", 
        "Capillary refill rate", 
        "Glascow coma scale verbal response",
        "Glascow coma scale eye opening"
    ]

    test_raw_data = load_data(test_reader, discretizer, normalizer, False)
    test_raw_data = remove_columns(test_raw_data, discretizer_header, columns_to_remove)
    
    return test_raw_data


def evaluate_model(model, test_data, model_type, device, num_mc_samples=50):
    """Evaluate model on test data and return metrics."""
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_epistemic = []
    all_aleatoric = []
    
    with torch.no_grad():
        for i in range(len(test_data[0])):
            x = test_data[0][i]
            y = test_data[1][i] if isinstance(test_data[1], list) else test_data[1]
            
            x = torch.FloatTensor(x).to(device)
            if x.dim() == 2:
                x = x.unsqueeze(0)
            
            if model_type == "probabilistic":
                mean, epistemic_var, aleatoric_var = model.predict_with_uncertainty(
                    x, num_samples=num_mc_samples
                )
                all_predictions.append(mean.cpu().numpy())
                all_epistemic.append(epistemic_var.cpu().numpy())
                all_aleatoric.append(aleatoric_var.cpu().numpy())
            else:
                output = model(x).view(-1)
                all_predictions.append(output.cpu().numpy())
            
            all_targets.append(y)
    
    # Convert to arrays
    predictions = np.array([p.flatten()[0] if hasattr(p, 'flatten') else p 
                           for p in all_predictions])
    targets = np.array(all_targets)
    
    # Calculate metrics
    binary_predictions = (predictions >= 0.5).astype(int)
    
    metrics = {
        'auroc': roc_auc_score(targets, predictions),
        'auprc': average_precision_score(targets, predictions),
        'precision': precision_score(targets, binary_predictions, zero_division=0),
        'recall': recall_score(targets, binary_predictions, zero_division=0),
        'f1': f1_score(targets, binary_predictions, zero_division=0),
        'accuracy': accuracy_score(targets, binary_predictions),
    }
    
    if model_type == "probabilistic":
        metrics['mean_epistemic'] = float(np.mean(all_epistemic))
        metrics['mean_aleatoric'] = float(np.mean(all_aleatoric))
    else:
        metrics['mean_epistemic'] = None
        metrics['mean_aleatoric'] = None
    
    return metrics


def save_results(metrics, model_name, model_type, eval_name, output_dir):
    """Save evaluation results to CSV and JSON."""
    # Create output directory
    results_dir = os.path.join(output_dir, f"eval_{eval_name.lower()}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Prepare display name
    if model_type == "probabilistic":
        model_display = f"{model_name.upper()} + MC Dropout"
        aleatoric = f"{metrics['mean_aleatoric']:.6f}" if metrics['mean_aleatoric'] else "–"
        epistemic = f"{metrics['mean_epistemic']:.6f}" if metrics['mean_epistemic'] else "–"
    else:
        model_display = model_name.upper()
        aleatoric = "–"
        epistemic = "–"
    
    results_data = {
        'Model': [model_display],
        'Dataset': [eval_name],
        'AUROC': [f"{metrics['auroc']:.4f}"],
        'AUPRC': [f"{metrics['auprc']:.4f}"],
        'Precision': [f"{metrics['precision']:.4f}"],
        'Recall': [f"{metrics['recall']:.4f}"],
        'F1-Score': [f"{metrics['f1']:.4f}"],
        'Aleatoric Uncertainty': [aleatoric],
        'Epistemic Uncertainty': [epistemic]
    }
    
    results_df = pd.DataFrame(results_data)
    
    # Save CSV
    csv_path = os.path.join(results_dir, 'results_table.csv')
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to: {csv_path}")
    
    # Save JSON with raw metrics
    json_path = os.path.join(results_dir, 'metrics.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to: {json_path}")
    
    # Print results
    print(f"\nEvaluation Results ({model_display} on {eval_name}):")
    print(results_df.to_string(index=False))
    
    return results_df


def load_config_from_hyperparams(hyperparams_path, model_name):
    """Load model config from best_hyperparams.txt or config_trial_X.json file."""
    config = {}
    
    # First try to load JSON config (preferred - more reliable)
    if hyperparams_path and hyperparams_path.endswith('.json') and os.path.exists(hyperparams_path):
        with open(hyperparams_path, 'r') as f:
            import json
            config = json.load(f)
        logger.info(f"Loaded config from JSON: {hyperparams_path}")
        return config
    
    # Try to find a config JSON file matching the model path
    if hyperparams_path:
        model_dir = os.path.dirname(hyperparams_path) if not os.path.isdir(hyperparams_path) else hyperparams_path
        # If we have a model file path, try to find corresponding config
        # e.g., final_model_trial_0.pth -> config_trial_0.json
        for f in os.listdir(model_dir) if os.path.exists(model_dir) else []:
            if f.startswith('config_trial_') and f.endswith('.json'):
                json_path = os.path.join(model_dir, f)
                with open(json_path, 'r') as jf:
                    import json
                    config = json.load(jf)
                logger.info(f"Loaded config from: {json_path}")
                return config
    
    # Fallback: try to load from best_hyperparams.txt
    if hyperparams_path and os.path.exists(hyperparams_path):
        with open(hyperparams_path, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Parse values
                    try:
                        if '.' in value:
                            config[key] = float(value)
                        else:
                            config[key] = int(value)
                    except ValueError:
                        config[key] = value
    
    # Set defaults for missing values
    defaults = {
        'input_size': 59,  # After removing columns
        'hidden_size': config.get('hidden_size', 64),
        'num_layers': config.get('num_layers', 2),
        'dropout': config.get('dropout', 0.2),
        'd_model': config.get('d_model', 64),
        'nhead': config.get('nhead', 4),
        'dim_feedforward': config.get('dim_feedforward', 256),
        'num_mc_samples': config.get('num_mc_samples', 50),
    }
    
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
    
    return config


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model on test data')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model file (.pth)')
    parser.add_argument('--model', type=str, required=True,
                       choices=['lstm', 'rnn', 'gru', 'transformer'],
                       help='Model architecture')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['deterministic', 'probabilistic'],
                       help='Model type')
    parser.add_argument('--eval_data', type=str, required=True,
                       help='Path to evaluation dataset')
    parser.add_argument('--eval_name', type=str, required=True,
                       help='Display name for evaluation dataset (e.g., INALO, MIMIC)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Base output directory for results')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Path to best_hyperparams.txt (optional)')
    parser.add_argument('--num_mc_samples', type=int, default=50,
                       help='Number of MC samples for probabilistic models')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load config - try to find config matching the specific model trial
    model_dir = os.path.dirname(args.model_path)
    model_filename = os.path.basename(args.model_path)
    
    # Extract trial number from model filename (e.g., final_model_trial_0.pth -> 0)
    import re
    trial_match = re.search(r'trial_(\d+)', model_filename)
    trial_num = trial_match.group(1) if trial_match else None
    
    config = None
    
    # First, try to load config for this specific trial
    if trial_num:
        trial_config_path = os.path.join(model_dir, f"config_trial_{trial_num}.json")
        if os.path.exists(trial_config_path):
            config = load_config_from_hyperparams(trial_config_path, args.model)
            logger.info(f"Loaded trial-specific config from: {trial_config_path}")
    
    # If no trial-specific config, try the provided config_path
    if config is None and args.config_path and os.path.exists(args.config_path):
        config = load_config_from_hyperparams(args.config_path, args.model)
    
    # Fallback: try to find any config in model directory
    if config is None:
        hyperparams_path = os.path.join(model_dir, 'best_hyperparams.txt')
        config = load_config_from_hyperparams(hyperparams_path, args.model)
    
    config['num_mc_samples'] = args.num_mc_samples
    logger.info(f"Model config: {config}")
    
    # Load model
    logger.info(f"Loading model from: {args.model_path}")
    model = load_model(args.model, args.model_type, args.model_path, config, device)
    
    # Load test data
    logger.info(f"Loading test data from: {args.eval_data}")
    test_data = load_test_data(args.eval_data)
    logger.info(f"Loaded {len(test_data[0])} test samples")
    
    # Evaluate
    logger.info("Evaluating model...")
    metrics = evaluate_model(model, test_data, args.model_type, device, 
                            config['num_mc_samples'])
    
    # Save results
    # Output structure: {output_dir}/{model_type}/{model}/eval_{eval_name}/
    full_output_dir = os.path.join(args.output_dir, args.model_type, args.model)
    save_results(metrics, args.model, args.model_type, args.eval_name, full_output_dir)
    
    logger.info("Evaluation complete!")
    return metrics


if __name__ == "__main__":
    main()
