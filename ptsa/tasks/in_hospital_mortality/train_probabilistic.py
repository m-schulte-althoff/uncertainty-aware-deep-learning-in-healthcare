import os
import logging
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import torch.nn.functional as F

import wandb
import optuna
import random

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, accuracy_score, precision_score, recall_score, roc_auc_score, f1_score

from ptsa.tasks.readers import InHospitalMortalityReader
from ptsa.utils.preprocessing import Discretizer, Normalizer
from ptsa.tasks.in_hospital_mortality.utils import load_data
from ptsa.utils import utils

from ptsa.models.probabilistic.lstm_classification import LSTM 
from ptsa.models.probabilistic.rnn_classification import RNN
from ptsa.models.probabilistic.gru_classification import GRU
from ptsa.models.probabilistic.transformer_classification import TransformerIHM 

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)


def remove_columns(data, discretizer_header, columns_to_remove):
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


def even_out_number_of_data_points(data):
    data_points, labels = data[0], data[1]
    logger.info("Number of Samples: %s", len(labels))
    positive_indices = [i for i, label in enumerate(labels) if label == 1]
    negative_indices = [i for i, label in enumerate(labels) if label == 0]
    
    logger.info("Number of Positive Samples: %s", len(positive_indices))
    logger.info("Number of Negative Samples: %s", len(negative_indices))
    target_size = min(len(positive_indices), len(negative_indices))

    sampled_positive_indices = random.sample(positive_indices, target_size)
    sampled_negative_indices = random.sample(negative_indices, target_size)

    balanced_indices = sampled_positive_indices + sampled_negative_indices
    random.shuffle(balanced_indices)

    balanced_data_points = [data_points[i] for i in balanced_indices]
    balanced_labels = [labels[i] for i in balanced_indices]
    logger.info("Number of Balanced Samples: %s", len(balanced_labels))
    return (balanced_data_points, balanced_labels)


def calculate_class_weights(labels):
    total_samples = len(labels)
    positive_samples = np.sum(labels)
    negative_samples = total_samples - positive_samples
    
    pos_weight = negative_samples / positive_samples
    
    return pos_weight


def log_detailed_metrics(targets, predictions, mean_epistemic=None, mean_aleatoric=None):
    targets = np.array(targets)
    predictions = np.array(predictions)
    
    binary_predictions = (predictions >= 0.5).astype(int)
    
    accuracy = accuracy_score(targets, binary_predictions)
    precision = precision_score(targets, binary_predictions)
    recall = recall_score(targets, binary_predictions)
    auc_roc = roc_auc_score(targets, predictions)
    f1 = f1_score(targets, binary_predictions)
    
    precisions, recalls, thresholds = precision_recall_curve(targets, predictions)
    avg_precision = average_precision_score(targets, predictions)
    
    wandb.log({
        "detailed_accuracy": accuracy,
        "detailed_precision": precision,
        "detailed_recall": recall,
        "detailed_auc_roc": auc_roc,
        "average_precision": avg_precision,
        "f1-score": f1,
        "average_precision": avg_precision,
        "f1-score": f1
    })
    
    plt.figure(figsize=(10, 6))
    plt.plot(recalls, precisions, label=f'AP={avg_precision:.2f}')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    
    wandb.log({"pr_curve": wandb.Image(plt)})
    plt.close()

    # Return all metrics as a dictionary
    return {
        'f1': f1,
        'auroc': auc_roc,
        'auprc': avg_precision,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'mean_epistemic': mean_epistemic,
        'mean_aleatoric': mean_aleatoric
    }

def binary_classification_uncertainty_loss(pred_proba, pred_log_var, targets, pos_weight):
            variance = torch.exp(pred_log_var)
            
            bce_loss = F.binary_cross_entropy(pred_proba, targets, 
                                            weight=pos_weight.expand_as(targets),
                                            reduction='none')
            
            uncertainty_reg = 0.5 * pred_log_var
            
            total_loss = (bce_loss / variance) + uncertainty_reg
            
            return total_loss.mean()


def binary_classification_uncertainty_loss_transformer(pred_proba, pred_log_var, targets, pos_weight):
    if len(pred_proba.shape) > 1:
        pred_proba = pred_proba.squeeze()
    if len(pred_log_var.shape) > 1:
        pred_log_var = pred_log_var.squeeze()
    if len(targets.shape) > 1:
        targets = targets.squeeze()
    
    pred_proba = torch.clamp(pred_proba, min=1e-6, max=1-1e-6)
    pred_log_var = torch.clamp(pred_log_var, min=-10, max=10)
    variance = torch.exp(pred_log_var)
    variance = torch.clamp(variance, min=1e-6, max=100)
    
    bce_loss = F.binary_cross_entropy(
        pred_proba, 
        targets,
        reduction='none'
    )
    
    if pos_weight is not None:
        weights = torch.where(targets == 1, pos_weight, torch.ones_like(pos_weight))
        weights = weights / weights.mean()
        bce_loss = bce_loss * weights
    
    uncertainty_loss = (bce_loss / variance) + 0.5 * torch.log(variance)
    
    uncertainty_loss = torch.clamp(uncertainty_loss, max=100)
    
    return uncertainty_loss.mean()

def objective(trial):
    wandb.finish()

    wandb.init(
        project=f"fixed_final_probabilistic_IHM", 
        group=f"final_{args.model}_classification",
        name=f"final_{args.model}_classification_trial_{trial.number}",
        reinit=True
    )
    try:
        config = {
            "input_size": 38,
            "hidden_size": trial.suggest_int('hidden_size', 32, 256),
            "num_layers": trial.suggest_int('num_layers', 1, 4),
            "learning_rate": trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
            "dropout": trial.suggest_float("dropout", 0.2, 0.8),
            "batch_size": trial.suggest_categorical('batch_size', [32, 64, 128]),
            "num_epochs": trial.suggest_int('num_epochs', 10, 40),
            "weight_decay": trial.suggest_loguniform("weight_decay", 1e-6, 1e-2),
            "num_mc_samples": 100
        }
        
        if args.model == "transformer":
            configurations = [
                {"d_model": 64, "nhead": 2},
                {"d_model": 64, "nhead": 4},
                {"d_model": 128, "nhead": 2},
                {"d_model": 128, "nhead": 4},
                {"d_model": 128, "nhead": 8},
                {"d_model": 256, "nhead": 4},
                {"d_model": 256, "nhead": 8}
            ]
            config_idx = trial.suggest_categorical("model_config", list(range(len(configurations))))
            selected_config = configurations[config_idx]
            
            config = {
                "input_size": 38,
                "d_model": selected_config["d_model"],
                "nhead": selected_config["nhead"],
                "num_layers": trial.suggest_int('num_layers', 1, 4),
                "dim_feedforward": trial.suggest_int("dim_feedforward", 64, 512, step=64),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True),
                "dropout": trial.suggest_float("dropout", 0.2, 0.8),
                "batch_size": trial.suggest_categorical('batch_size', [32, 64, 128]),
                "num_mc_samples": 100,
                "weight_decay": trial.suggest_loguniform("weight_decay", 1e-4, 1e-2),
                "num_epochs": trial.suggest_int('num_epochs', 5, 15),
            }

        wandb.config.update(config)

        device = "cuda:3" if torch.cuda.is_available() else "cpu"

        all_reader = InHospitalMortalityReader(
            dataset_dir=os.path.join(args.data, 'train'), 
            listfile=os.path.join(args.data, 'train/listfile.csv'), 
            period_length=48.0
        )

        train_data, val_data = train_test_split(all_reader._data, test_size=0.2, random_state=43)

        train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'), listfile=os.path.join(args.data, 'train/listfile.csv'), period_length=48.0)
        train_reader._data = train_data
        
        val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'), listfile=os.path.join(args.data, 'train/listfile.csv'), period_length=48.0)
        val_reader._data = val_data

        test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'), listfile=os.path.join(args.data, 'test/listfile.csv'), period_length=48.0)

        discretizer = Discretizer(
            timestep=float(args.timestep), 
            store_masks=True, 
            impute_strategy='previous', 
            start_time='zero'
        )
        discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
        cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

        normalizer = Normalizer(fields=cont_channels)
        normalizer_state = args.normalizer_state
        if normalizer_state is None:
            normalizer_state = f'ihm_ts{args.timestep}.input_str_{args.imputation}.start_time_zero.normalizer'
            normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
        normalizer.load_params(normalizer_state)
        
        columns_to_remove = [
            "Glascow coma scale motor response", 
            "Capillary refill rate", 
            "Glascow coma scale verbal response",
            "Glascow coma scale eye opening"

        ]

        train_raw_data = load_data(train_reader, discretizer, normalizer, args.small_part)
        val_raw_data = load_data(val_reader, discretizer, normalizer, args.small_part)
        test_raw_data = load_data(test_reader, discretizer, normalizer, args.small_part)

        train_raw_data = remove_columns(train_raw_data, discretizer_header, columns_to_remove)
        val_raw_data = remove_columns(val_raw_data, discretizer_header, columns_to_remove)
        test_raw_data = remove_columns(test_raw_data, discretizer_header, columns_to_remove)

        train_raw = even_out_number_of_data_points(train_raw_data)
        val_raw = even_out_number_of_data_points(val_raw_data)
        test_raw = even_out_number_of_data_points(test_raw_data)

        train_labels = train_raw[1]
        pos_weight = calculate_class_weights(train_labels)
        logger.info("Pos Weight: %s", pos_weight)

        wandb.log({"pos_class_weight": pos_weight})

        model = nn.Module()
        if args.model == "lstm":
            model = LSTM(config["input_size"], config["hidden_size"], config["num_layers"], config["dropout"]).to(device)
        elif args.model == "rnn":
            model = RNN(config["input_size"], config["hidden_size"], config["num_layers"], config["dropout"]).to(device)
        elif args.model == "gru":
            model = GRU(config["input_size"], config["hidden_size"], config["num_layers"], config["dropout"]).to(device)
        elif args.model == "transformer":
            model = TransformerIHM(input_size=config["input_size"],
                                d_model=config["d_model"],
                                nhead=config["nhead"],
                                num_layers=config["num_layers"],
                                dropout=config["dropout"],
                                dim_feedforward=config["dim_feedforward"]).to(device)


        pos_weight_tensor = torch.tensor([pos_weight], device=device)
        criterion = nn.BCELoss(weight=pos_weight_tensor)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        if args.model == "transformer":
            optimizer = torch.optim.AdamW(
                                        model.parameters(),
                                        lr=config["learning_rate"],
                                        weight_decay=config["weight_decay"],
                                        eps=1e-8
                                        )

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                                                optimizer,
                                                                mode='min',
                                                                factor=0.5,
                                                                patience=3,
                                                                min_lr=1e-6
                                                                )

        for epoch in range(config["num_epochs"]):
            model.train()
            train_loss = 0
            for i in range(len(train_raw[0])):
                x, y = train_raw[0][i], train_raw[1]
                x = torch.FloatTensor(x).to(device)
                y = torch.FloatTensor([y[i] if isinstance(y, (list, np.ndarray)) else y]).to(device)

                if x.dim() == 2:
                    x = x.unsqueeze(0)
                
                optimizer.zero_grad()

                mean, log_var = model(x)
                    
                if args.model == "transformer":
                    if torch.any(torch.isnan(mean)) or torch.any(torch.isnan(log_var)):
                        print(f"Warning: NaN detected in model outputs")
                        continue

                    loss = binary_classification_uncertainty_loss_transformer(
                        mean,
                        log_var, 
                        y,
                        pos_weight_tensor
                    )

                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"Warning: Invalid loss value: {loss}")
                        print(f"Probabilities range: [{mean.min()}, {mean.max()}]")
                        print(f"Log variance range: [{log_var.min()}, {log_var.max()}]")
                        continue
                
                else:
                    loss = binary_classification_uncertainty_loss(
                        mean,
                        log_var, 
                        y,
                        pos_weight_tensor
                    )
 
                loss.backward()

                if args.model == "transformer":
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_raw[0])

            
            model.eval()
            val_loss = 0
            predictions_val = []
            targets_val = []
            for i in range(len(val_raw[0])):
                x, y = val_raw[0][i], val_raw[1]
                x = torch.FloatTensor(x).to(device)
                y = torch.FloatTensor([y[i] if isinstance(y, (list, np.ndarray)) else y]).to(device)
                
                if x.dim() == 2:
                    x = x.unsqueeze(0)
                
                mean, log_var = model(x)
                
                if args.model == "transformer":
                    loss = binary_classification_uncertainty_loss_transformer(
                        mean,
                        log_var, 
                        y,
                        pos_weight_tensor
                    )
                else:
                    loss = binary_classification_uncertainty_loss(
                    mean,
                    log_var, 
                    y,
                    pos_weight_tensor
                )

                
                val_loss += loss.item()

                predictions_val.append(mean.detach().cpu().numpy())
                targets_val.append(y.cpu().numpy())

            val_loss /= len(val_raw[0])

            if args.model == "transformer":
                scheduler.step(val_loss)

            predictions = [pred[0] for pred in predictions_val]
            targets = [target[0] for target in targets_val]
            
            binary_predictions = (np.array(predictions) >= 0.5).astype(int)
            f1 = f1_score(targets, binary_predictions)

            wandb.log({
                "epoch": epoch,
                "val_loss": val_loss,
                "f1": f1,
            })

            trial.report(f1, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss
            })

        model.eval()
        all_predictions = []
        all_targets = []
        all_epistemic = []
        all_aleatoric = []
        
        with torch.no_grad():
            for i in range(len(test_raw[0])):
                x, y = test_raw[0][i], test_raw[1]
                x = torch.FloatTensor(x).to(device)
                y = torch.FloatTensor([y[i] if isinstance(y, (list, np.ndarray)) else y]).to(device)

                if x.dim() == 2:
                    x = x.unsqueeze(0)
                
                # outputs = model(x).view(-1)

                mean, epistemic_var, aleatoric_var = model.predict_with_uncertainty(x, num_samples=config["num_mc_samples"])
                
                all_predictions.append(mean.cpu().numpy())
                all_epistemic.append(epistemic_var.cpu().numpy())
                all_aleatoric.append(aleatoric_var.cpu().numpy())
                all_targets.append(y.cpu().numpy())

        predictions = [pred for pred in all_predictions]
        targets = [target[0] for target in all_targets]
        
        mean_epistemic = np.mean(all_epistemic)
        mean_aleatoric = np.mean(all_aleatoric)
        
        metrics = log_detailed_metrics(targets, predictions, mean_epistemic, mean_aleatoric)
        
        model_path = os.path.join(args.output_dir, f"final_model_trial_{trial.number}.pth")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        logger.info("Saving final model to: %s", model_path)
        torch.save(model.state_dict(), model_path)
        
        # Save model config for this trial
        config_path = os.path.join(args.output_dir, f"config_trial_{trial.number}.json")
        import json
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info("Saving config to: %s", config_path)
        
        artifact = wandb.Artifact(
                    name=f'model-trial-{trial.number}',
                    type='model',
                    description=f'Best model for trial {trial.number} with F1={metrics["f1"]:.4f}'
                )
        artifact.add_file(model_path, f"final_model_trial_{trial.number}.pth")
        artifact.save()
        # run.log_artifact(artifact)

        wandb.log({
            "Mean Epistemic Uncertainty": mean_epistemic,
            "Mean Aleatoric Uncertainty": mean_aleatoric
        })
        
        # Store metrics in trial user attributes for later retrieval
        trial.set_user_attr('auroc', metrics['auroc'])
        trial.set_user_attr('auprc', metrics['auprc'])
        trial.set_user_attr('precision', metrics['precision'])
        trial.set_user_attr('recall', metrics['recall'])
        trial.set_user_attr('f1', metrics['f1'])
        trial.set_user_attr('mean_epistemic', mean_epistemic)
        trial.set_user_attr('mean_aleatoric', mean_aleatoric)

        auc_roc = roc_auc_score(targets, predictions)

        return metrics['f1']
    
    except RuntimeError as e:
        print(f"Error in training step:")
        print(f"Input shape: {x.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Mean output shape: {mean.shape if 'mean' in locals() else 'Not computed'}")
        print(f"Log var shape: {log_var.shape if 'log_var' in locals() else 'Not computed'}")
        raise e

    finally:
        wandb.finish()

def main():
    parser = argparse.ArgumentParser()
    utils.add_common_arguments(parser)
    parser.add_argument('--data', type=str, 
        help='Path to the data of in-hospital mortality task', 
        default=os.path.join(os.path.dirname(__file__), '../../data/in-hospital-mortality/')
    )
    parser.add_argument('--output_dir', type=str, 
        help='Directory relative which all output files are stored', 
        default='.'
    )
    parser.add_argument('--num_trials', type=int, default=10, 
        help='Number of Optuna trials to run')
    
    parser.add_argument('--model', type=str, default="lstm", help="lstm, rnn, gru, transformer")
    parser.add_argument("--model_name", type=str, help="Name for the model file")


    parser.add_argument('--partition', type=str, default='custom',
                    help="log, custom, none")

    global args
    args = parser.parse_args()
    
    # Extract dataset name from data path for output organization
    dataset_name = os.path.basename(os.path.normpath(args.data))
    
    # Build output directory structure: {output_dir}/{dataset}/{probabilistic}/{model}/
    args.output_dir = os.path.join(args.output_dir, dataset_name, "probabilistic", args.model)
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")

    study = optuna.create_study(
        direction='maximize', 
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=6)
    )
    
    # study = optuna.create_study(direction="maximize")

    study.optimize(objective, n_trials=args.num_trials)

    print('Number of finished trials:', len(study.trials))
    print('Best trial:')
    trial = study.best_trial

    print('  Value (AUC-ROC): ', trial.value)
    print('  Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')

    with open(os.path.join(args.output_dir, 'best_hyperparams.txt'), 'w') as f:
        f.write(f"Best AUC-ROC: {trial.value}\n")
        for key, value in trial.params.items():
            f.write(f"{key}: {value}\n")

    # Determine dataset display name
    dataset_display = "MIMIC" if "fixed" in dataset_name else "INALO"
    
    # Create results table
    results_data = {
        'Model': [f"{args.model.upper()} + MC Dropout"],
        'Dataset': [dataset_display],
        'AUROC': [f"{trial.user_attrs.get('auroc', 0.0):.4f}"],
        'AUPRC': [f"{trial.user_attrs.get('auprc', 0.0):.4f}"],
        'Precision': [f"{trial.user_attrs.get('precision', 0.0):.4f}"],
        'Recall': [f"{trial.user_attrs.get('recall', 0.0):.4f}"],
        'F1-Score': [f"{trial.user_attrs.get('f1', 0.0):.4f}"],
        'Aleatoric Uncertainty': [f"{trial.user_attrs.get('mean_aleatoric', 0.0):.6f}"],
        'Epistemic Uncertainty': [f"{trial.user_attrs.get('mean_epistemic', 0.0):.6f}"]
    }
    
    results_df = pd.DataFrame(results_data)
    
    # Save as CSV
    csv_path = os.path.join(args.output_dir, 'results_table.csv')
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Results table saved to: {csv_path}")
    
    # Save as formatted text table
    txt_path = os.path.join(args.output_dir, 'results_table.txt')
    with open(txt_path, 'w') as f:
        f.write(results_df.to_string(index=False))
    logger.info(f"Results table (txt) saved to: {txt_path}")
    
    # Print the table
    print("\nFinal Results:")
    print(results_df.to_string(index=False))

    # Generate visualizations (may fail with single trial)
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image(os.path.join(args.output_dir, 'optimization_history.png'))
    except Exception as e:
        logger.warning(f"Could not generate optimization history plot: {e}")

    try:
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_image(os.path.join(args.output_dir, 'param_importances.png'))
    except Exception as e:
        logger.warning(f"Could not generate param importances plot: {e}")

if __name__ == "__main__":
    main()

