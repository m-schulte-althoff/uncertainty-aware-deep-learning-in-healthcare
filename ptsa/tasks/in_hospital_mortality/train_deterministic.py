from operator import neg
import os
import logging
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn

import wandb
import optuna
import random

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, accuracy_score, precision_score, recall_score, roc_auc_score, f1_score

from ptsa.tasks.readers import InHospitalMortalityReader
from ptsa.utils.preprocessing import Discretizer, Normalizer
from ptsa.tasks.in_hospital_mortality.utils import load_data
from ptsa.utils import utils

from ptsa.models.deterministic.lstm_classification import LSTM 
from ptsa.models.deterministic.rnn_classification import RNN
from ptsa.models.deterministic.gru_classification import GRU
from ptsa.models.deterministic.transformer_classification import TransformerIHM

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

best_f1_score = 0.0

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

def calculate_class_weights(labels):
    total_samples = len(labels)
    positive_samples = np.sum(labels)
    negative_samples = total_samples - positive_samples
    
    pos_weight = negative_samples / positive_samples
    
    return pos_weight


def log_detailed_metrics(targets, predictions):
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

    return f1


def objective(trial):
    wandb.finish()

    wandb.init(
        project="fixed_final_deterministic_IHM", 
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
            "dropout": 0.2,
            "batch_size": trial.suggest_categorical('batch_size', [32, 64, 128]),
            "num_epochs": trial.suggest_int('num_epochs', 10, 40),
            "weight_decay": trial.suggest_loguniform("weight_decay", 1e-6, 1e-2),
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
                outputs = model(x).view(-1)
                loss = criterion(outputs, y)
                loss.backward()
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
                
                outputs = model(x)

                outputs = outputs.view(-1)
                loss = criterion(outputs, y)
                val_loss += loss.item()

                predictions_val.append(outputs.detach().cpu().numpy())
                targets_val.append(y.cpu().numpy())

            val_loss /= len(val_raw[0])

            """
            # Compute F1 score and best threshold
            predictions = [pred[0] for pred in predictions_val]
            targets = [target[0] for target in targets_val]
            
            thresholds = np.linspace(0, 1, 100)
            best_f1, best_thresh = 0, 0
            for thresh in thresholds:
                binary_predictions = (np.array(predictions) >= thresh).astype(int)
                f1 = f1_score(targets, binary_predictions)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh

                    torch.save(model.state_dict(), os.path.join(args.output_dir, f"{args.model}/best_model_epoch{epoch}.pth"))
            """

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
        
        with torch.no_grad():
            for i in range(len(test_raw[0])):
                x, y = test_raw[0][i], test_raw[1]
                x = torch.FloatTensor(x).to(device)
                y = torch.FloatTensor([y[i] if isinstance(y, (list, np.ndarray)) else y]).to(device)

                if x.dim() == 2:
                    x = x.unsqueeze(0)
                
                outputs = model(x).view(-1)

                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(y.cpu().numpy())
        
        predictions = [pred[0] for pred in all_predictions]
        targets = [target[0] for target in all_targets]
        
        f1_score_testing = log_detailed_metrics(targets, predictions)
        
        model_path = os.path.join(args.output_dir, f"{args.model}/final_model_trial_{trial.number}.pth")
        logger.info("Saving final model to: %s", model_path)
        torch.save(model.state_dict(), model_path)
        
        artifact = wandb.Artifact(
                    name=f'model-trial-{trial.number}',
                    type='model',
                    description=f'Best model for trial {trial.number} with F1={f1_score_testing:.4f}'
                )
        artifact.add_file(model_path, f"final_model_trial_{trial.number}.pth")
        artifact.save()
        # run.log_artifact(artifact)

        

        auc_roc = roc_auc_score(targets, predictions)

        return f1_score_testing
    
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

    optuna.visualization.plot_optimization_history(study)
    plt.savefig(os.path.join(args.output_dir, 'optimization_history.png'))
    optuna.visualization.plot_param_importances(study)
    plt.savefig(os.path.join(args.output_dir, 'param_importances.png'))

if __name__ == "__main__":
    main()

