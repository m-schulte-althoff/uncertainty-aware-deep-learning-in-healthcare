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


def remove_columns(data, discretizer_header, columns_to_remove):
    """Remove specified columns from the data.
    
    This matches the training script behavior exactly:
    - Removes all columns where the column name CONTAINS any of the removal strings
    - This includes one-hot encoded variants and mask columns
    """
    data_points, labels = data
    
    indices_to_remove = []
    for col in columns_to_remove:
        for i, header in enumerate(discretizer_header):
            if col in header:
                indices_to_remove.append(i)
    
    modified_data_points = []
    for patient_data in data_points:
        filtered_data = np.delete(patient_data, indices_to_remove, axis=1)
        modified_data_points.append(filtered_data)

    return (modified_data_points, labels)


def infer_config_from_state_dict(state_dict, model_name, model_type):
    """Infer model configuration from the saved state dict weights.
    
    This is the most reliable way to load a model since we read the actual
    weight shapes instead of relying on config files that may not match.
    """
    config = {'dropout': 0.2}  # dropout doesn't affect weight shapes
    
    if model_name == "lstm":
        # LSTM weight_ih_l0 shape is (4*hidden_size, input_size)
        weight_ih_l0 = state_dict['lstm.weight_ih_l0']
        hidden_size = weight_ih_l0.shape[0] // 4
        input_size = weight_ih_l0.shape[1]
        # Count layers by counting weight_ih_lX keys
        num_layers = sum(1 for k in state_dict.keys() if k.startswith('lstm.weight_ih_l'))
        config.update({
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers
        })
        
    elif model_name == "rnn":
        # RNN weight_ih_l0 shape is (hidden_size, input_size)
        weight_ih_l0 = state_dict['rnn.weight_ih_l0']
        hidden_size = weight_ih_l0.shape[0]
        input_size = weight_ih_l0.shape[1]
        num_layers = sum(1 for k in state_dict.keys() if k.startswith('rnn.weight_ih_l'))
        config.update({
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers
        })
        
    elif model_name == "gru":
        # GRU weight_ih_l0 shape is (3*hidden_size, input_size)
        weight_ih_l0 = state_dict['gru.weight_ih_l0']
        hidden_size = weight_ih_l0.shape[0] // 3
        input_size = weight_ih_l0.shape[1]
        num_layers = sum(1 for k in state_dict.keys() if k.startswith('gru.weight_ih_l'))
        config.update({
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers
        })
        
    elif model_name == "transformer":
        # Transformer: infer from input_projection and transformer layers
        input_proj = state_dict['input_projection.weight']
        d_model = input_proj.shape[0]
        input_size = input_proj.shape[1]
        
        # Count transformer layers
        layer_indices = set()
        for k in state_dict.keys():
            if k.startswith('transformer_encoder.layers.'):
                parts = k.split('.')
                layer_indices.add(int(parts[2]))
        num_layers = len(layer_indices)
        
        # Infer dim_feedforward from linear1 weight shape
        linear1_weight = state_dict['transformer_encoder.layers.0.linear1.weight']
        dim_feedforward = linear1_weight.shape[0]
        
        # Infer nhead from attention weights
        # in_proj_weight shape is (3*d_model, d_model) for self-attention
        in_proj = state_dict['transformer_encoder.layers.0.self_attn.in_proj_weight']
        # nhead is typically d_model // head_dim, we'll use common values
        # Check if d_model is divisible by common head counts
        for nhead in [8, 4, 2, 1]:
            if d_model % nhead == 0:
                break
        
        config.update({
            'input_size': input_size,
            'd_model': d_model,
            'num_layers': num_layers,
            'dim_feedforward': dim_feedforward,
            'nhead': nhead
        })
    
    return config


def load_model(model_name, model_type, model_path, device):
    """Load a trained model.
    
    The config is inferred from the state dict, so it always matches
    the actual saved weights.
    """
    # Load state dict first to infer config
    state_dict = torch.load(model_path, weights_only=True, map_location=device)
    inferred_config = infer_config_from_state_dict(state_dict, model_name, model_type)
    
    logger.info(f"Inferred config from checkpoint: {inferred_config}")
    
    if model_type == "deterministic":
        if model_name == "lstm":
            from ptsa.models.deterministic.lstm_classification import LSTM
            model = LSTM(inferred_config["input_size"], inferred_config["hidden_size"], 
                        inferred_config["num_layers"], inferred_config.get("dropout", 0.2)).to(device)
        elif model_name == "rnn":
            from ptsa.models.deterministic.rnn_classification import RNN
            model = RNN(inferred_config["input_size"], inferred_config["hidden_size"], 
                       inferred_config["num_layers"], inferred_config.get("dropout", 0.2)).to(device)
        elif model_name == "gru":
            from ptsa.models.deterministic.gru_classification import GRU
            model = GRU(inferred_config["input_size"], inferred_config["hidden_size"], 
                       inferred_config["num_layers"], inferred_config.get("dropout", 0.2)).to(device)
        elif model_name == "transformer":
            from ptsa.models.deterministic.transformer_classification import TransformerIHM
            model = TransformerIHM(
                input_size=inferred_config["input_size"],
                d_model=inferred_config["d_model"],
                nhead=inferred_config["nhead"],
                num_layers=inferred_config["num_layers"],
                dropout=inferred_config.get("dropout", 0.2),
                dim_feedforward=inferred_config["dim_feedforward"]
            ).to(device)
    else:  # probabilistic
        if model_name == "lstm":
            from ptsa.models.probabilistic.lstm_classification import LSTM
            model = LSTM(inferred_config["input_size"], inferred_config["hidden_size"], 
                        inferred_config["num_layers"], inferred_config.get("dropout", 0.2)).to(device)
        elif model_name == "rnn":
            from ptsa.models.probabilistic.rnn_classification import RNN
            model = RNN(inferred_config["input_size"], inferred_config["hidden_size"], 
                       inferred_config["num_layers"], inferred_config.get("dropout", 0.2)).to(device)
        elif model_name == "gru":
            from ptsa.models.probabilistic.gru_classification import GRU
            model = GRU(inferred_config["input_size"], inferred_config["hidden_size"], 
                       inferred_config["num_layers"], inferred_config.get("dropout", 0.2)).to(device)
        elif model_name == "transformer":
            from ptsa.models.probabilistic.transformer_classification import TransformerIHM
            model = TransformerIHM(
                input_size=inferred_config["input_size"],
                d_model=inferred_config["d_model"],
                nhead=inferred_config["nhead"],
                num_layers=inferred_config["num_layers"],
                dropout=inferred_config.get("dropout", 0.2),
                dim_feedforward=inferred_config["dim_feedforward"]
            ).to(device)
    
    model.load_state_dict(state_dict)
    model.eval()
    return model, inferred_config


def load_test_data(data_path, expected_input_size=None, timestep=1.0):
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
    
    # Convert to padded batch format for faster processing
    data_points, labels = test_raw_data
    
    # Find max sequence length and pad
    max_len = max(x.shape[0] for x in data_points)
    num_features = data_points[0].shape[1]
    
    logger.info(f"Data has {num_features} features, max sequence length {max_len}")
    
    # Check if we need to adjust feature count to match model
    if expected_input_size is not None and num_features != expected_input_size:
        logger.warning(f"Feature mismatch: data has {num_features} features, model expects {expected_input_size}")
        if num_features > expected_input_size:
            # Truncate features
            data_points = [x[:, :expected_input_size] for x in data_points]
            num_features = expected_input_size
            logger.info(f"Truncated data to {num_features} features")
        else:
            # Pad features with zeros
            padded_data_points = []
            for x in data_points:
                padded = np.zeros((x.shape[0], expected_input_size))
                padded[:, :num_features] = x
                padded_data_points.append(padded)
            data_points = padded_data_points
            num_features = expected_input_size
            logger.info(f"Padded data to {num_features} features")
    
    # Pad sequences to same length
    padded_data = np.zeros((len(data_points), max_len, num_features), dtype=np.float32)
    for i, x in enumerate(data_points):
        padded_data[i, :x.shape[0], :] = x
    
    labels = np.array(labels, dtype=np.float32)
    
    return padded_data, labels


def evaluate_deterministic_batched(model, data, labels, device, batch_size=256):
    """Evaluate a deterministic model using batched inference."""
    model.eval()
    
    all_predictions = []
    n_samples = len(data)
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_data = data[i:i+batch_size]
            x = torch.FloatTensor(batch_data).to(device)
            
            output = model(x)
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(output).cpu().numpy().flatten()
            all_predictions.extend(probs)
    
    predictions = np.array(all_predictions)
    
    # Calculate metrics
    binary_predictions = (predictions >= 0.5).astype(int)
    
    metrics = {
        'auroc': roc_auc_score(labels, predictions),
        'auprc': average_precision_score(labels, predictions),
        'precision': precision_score(labels, binary_predictions, zero_division=0),
        'recall': recall_score(labels, binary_predictions, zero_division=0),
        'f1': f1_score(labels, binary_predictions, zero_division=0),
        'accuracy': accuracy_score(labels, binary_predictions),
        'mean_epistemic': None,
        'mean_aleatoric': None,
    }
    
    return metrics


def evaluate_probabilistic_batched(model, data, labels, device, num_mc_samples=50, batch_size=128):
    """Evaluate a probabilistic model using batched MC Dropout inference."""
    model.train()  # Keep dropout enabled for MC sampling
    
    n_samples = len(data)
    all_mean_preds = []
    all_epistemic = []
    all_aleatoric = []
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_data = data[i:i+batch_size]
            x = torch.FloatTensor(batch_data).to(device)
            
            # MC Dropout: run multiple forward passes
            mc_outputs = []
            mc_log_vars = []
            
            for _ in range(num_mc_samples):
                output, log_var = model(x)
                mc_outputs.append(torch.sigmoid(output))
                mc_log_vars.append(log_var)
            
            # Stack MC samples: (num_mc_samples, batch_size, 1)
            mc_outputs = torch.stack(mc_outputs, dim=0)
            mc_log_vars = torch.stack(mc_log_vars, dim=0)
            
            # Mean prediction across MC samples
            mean_pred = mc_outputs.mean(dim=0).cpu().numpy().flatten()
            
            # Epistemic uncertainty: variance of predictions across MC samples
            epistemic = mc_outputs.var(dim=0).cpu().numpy().flatten()
            
            # Aleatoric uncertainty: mean of predicted variances
            aleatoric = torch.exp(mc_log_vars).mean(dim=0).cpu().numpy().flatten()
            
            all_mean_preds.extend(mean_pred)
            all_epistemic.extend(epistemic)
            all_aleatoric.extend(aleatoric)
    
    predictions = np.array(all_mean_preds)
    epistemic_arr = np.array(all_epistemic)
    aleatoric_arr = np.array(all_aleatoric)
    
    # Calculate metrics
    binary_predictions = (predictions >= 0.5).astype(int)
    
    metrics = {
        'auroc': roc_auc_score(labels, predictions),
        'auprc': average_precision_score(labels, predictions),
        'precision': precision_score(labels, binary_predictions, zero_division=0),
        'recall': recall_score(labels, binary_predictions, zero_division=0),
        'f1': f1_score(labels, binary_predictions, zero_division=0),
        'accuracy': accuracy_score(labels, binary_predictions),
        'mean_epistemic': float(np.mean(epistemic_arr)),
        'mean_aleatoric': float(np.mean(aleatoric_arr)),
    }
    
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
    parser.add_argument('--num_mc_samples', type=int, default=50,
                       help='Number of MC samples for probabilistic models')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model (config is inferred from checkpoint)
    logger.info(f"Loading model from: {args.model_path}")
    model, config = load_model(args.model, args.model_type, args.model_path, device)
    
    # Load test data, passing expected input size for compatibility check
    logger.info(f"Loading test data from: {args.eval_data}")
    test_data, test_labels = load_test_data(
        args.eval_data, 
        expected_input_size=config.get('input_size')
    )
    logger.info(f"Loaded {len(test_labels)} test samples, shape: {test_data.shape}")
    
    # Evaluate
    logger.info("Evaluating model...")
    if args.model_type == "deterministic":
        metrics = evaluate_deterministic_batched(model, test_data, test_labels, device)
    else:
        metrics = evaluate_probabilistic_batched(
            model, test_data, test_labels, device, 
            num_mc_samples=args.num_mc_samples
        )
    
    # Save results
    # Output structure: {output_dir}/{model_type}/{model}/eval_{eval_name}/
    full_output_dir = os.path.join(args.output_dir, args.model_type, args.model)
    save_results(metrics, args.model, args.model_type, args.eval_name, full_output_dir)
    
    logger.info("Evaluation complete!")
    return metrics


if __name__ == "__main__":
    main()
