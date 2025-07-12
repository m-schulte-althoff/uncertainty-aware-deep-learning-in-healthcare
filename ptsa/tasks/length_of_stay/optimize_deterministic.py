import os
import argparse
import random
import math

import torch
from torch import nn
from sklearn.model_selection import train_test_split
import wandb
import numpy as np
import optuna
import matplotlib.pyplot as plt

from ptsa.tasks.readers import LengthOfStayReader
from ptsa.tasks.length_of_stay.utils import BatchGen
from ptsa.utils.preprocessing import Normalizer, Discretizer
from ptsa.tasks.length_of_stay.utils import utils

from ptsa.models.deterministic.lstm import LSTM 
from ptsa.models.deterministic.rnn import RNN
from ptsa.models.deterministic.gru import GRU
from ptsa.models.deterministic.transformer import TransformerLOS

# EXAMPLE USAGE
# python ptsa/tasks/length_of_stay/train_probabilistic_lstm.py --data /vol/tmp/scholuka/mimic-iv-benchmarks/data/length-of-stay --network ptsa/models/probabilistic/gru.py --model_name test_gru --model gru


def get_random_slice(data_length, batch_size, target_fraction=0.5):
    target_size = math.floor(data_length * target_fraction)
    adjusted_size = (target_size // batch_size) * batch_size

    slice_size = max(adjusted_size, batch_size)

    max_start = data_length - slice_size

    valid_starts = list(range(0, max_start + 1, batch_size))
    if not valid_starts:
        return 0, batch_size
    
    start_idx = random.choice(valid_starts)
    end_idx = start_idx + slice_size
    
    return start_idx, end_idx

def nll_loss(mean, log_var, y):
    variance = torch.exp(log_var)
    return 0.5 * torch.mean((y - mean) ** 2 / variance + log_var)

def objective(trial):
    wandb.finish()
    wandb.init(
        project=f"fixed_final_deterministic_length_of_stay", 
        group=f"final_{args.model}",
        name=f"final_{args.model}_trial_{trial.number}",
        reinit=True
    )
    try:

        config = {
            "input_size": 38,
            "hidden_size": trial.suggest_int('hidden_size', 32, 256),
            "num_layers": trial.suggest_int('num_layers', 1, 4),
            "learning_rate": trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
            "num_epochs": trial.suggest_int('num_epochs', 5, 15),
            "batch_size": trial.suggest_categorical('batch_size', [32, 64, 128]),
            "dropout": trial.suggest_float("dropout", 0.2, 0.8),
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
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                "dropout": trial.suggest_float("dropout", 0.2, 0.8),
                "batch_size": trial.suggest_categorical('batch_size', [32, 64, 128]),
                "num_mc_samples": 100,
                "num_epochs": trial.suggest_int('num_epochs', 5, 15),
            }

        wandb.config.update(config)
        
        device = "cuda:2" if torch.cuda.is_available() else "cpu" 

        model = nn.Module()
        if args.model == "lstm":
            model = LSTM(config["input_size"], config["hidden_size"], config["num_layers"], config["dropout"]).to(device)
        elif args.model == "rnn":
            model = RNN(config["input_size"], config["hidden_size"], config["num_layers"], config["dropout"]).to(device)
        elif args.model == "gru": 
            model = GRU(config["input_size"], config["hidden_size"], config["num_layers"], config["dropout"]).to(device)
        elif args.model == "transformer":
            model = TransformerLOS(input_size=config["input_size"],
                                d_model=config["d_model"],
                                nhead=config["nhead"],
                                num_layers=config["num_layers"],
                                dropout=config["dropout"],
                                dim_feedforward=config["dim_feedforward"]).to(device)

        print(f"Model device: {next(model.parameters()).device}")

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

        all_data = LengthOfStayReader(dataset_dir=os.path.join(args.data, 'train'),
                                        listfile=os.path.join(args.data, 'train/listfile.csv'))

        train_data, val_data = train_test_split(all_data._data, test_size=0.2, random_state=42)
        # train_val_data, test_data = train_test_split(all_data._data, test_size=0.2, random_state=42)
        # train_data, val_data = train_test_split(train_val_data, test_size=0.2, random_state=42)

        if args.num_train_samples is not None:
            train_data = train_data[:args.num_train_samples]

        train_reader = LengthOfStayReader(dataset_dir=os.path.join(args.data, 'train'))
        train_reader._data = train_data

        if args.dataset_fraction:
            start_idx, end_idx = get_random_slice(
                data_length=len(train_reader._data),
                batch_size=config["batch_size"],
                target_fraction=args.dataset_fraction
            )
            train_reader._data = train_reader._data[start_idx:end_idx]

        val_reader = LengthOfStayReader(dataset_dir=os.path.join(args.data, 'train'))
        val_reader._data = val_data

        if args.dataset_fraction:
            start_idx, end_idx = get_random_slice(
                data_length=len(val_reader._data),
                batch_size=config["batch_size"],
                target_fraction=args.dataset_fraction
            )
            val_reader._data = val_reader._data[start_idx:end_idx]


        test_reader = LengthOfStayReader(dataset_dir=os.path.join(args.data, "test"),
                                        listfile=os.path.join(args.data, "test/listfile.csv"))

        if args.dataset_fraction:
            start_idx, end_idx = get_random_slice(
                data_length=len(test_reader._data),
                batch_size=config["batch_size"],
                target_fraction=args.dataset_fraction
            )
            test_reader._data = test_reader._data[start_idx:end_idx]

        # test_reader = LengthOfStayReader(dataset_dir=os.path.join(args.data, 'train'))
        # test_reader._data = test_data

        discretizer = Discretizer(timestep=args.timestep,
                                    store_masks=True,
                                    impute_strategy='previous',
                                    start_time='zero')

        discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
        cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

        normalizer = Normalizer(fields=cont_channels)
        normalizer_state = args.normalizer_state

        if normalizer_state is None:
            normalizer_state = 'los_ts{}.input_str_previous.start_time_zero.n5e4.normalizer'.format(args.timestep)
            normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)

        normalizer.load_params(normalizer_state)

        columns_to_drop = [
            "Glascow coma scale motor response", 
            "Capillary refill rate", 
            "Glascow coma scale verbal response",
            "Glascow coma scale eye opening"
        ]

        train_data_gen = BatchGen(reader=train_reader,
                                    discretizer=discretizer,
                                    normalizer=normalizer,
                                    batch_size=config["batch_size"],
                                    steps=None,
                                    shuffle=True,
                                    partition=args.partition,
                                    columns_to_drop=columns_to_drop)

        val_data_gen = BatchGen(reader=val_reader,
                                discretizer=discretizer,
                                normalizer=normalizer,
                                batch_size=config["batch_size"],
                                steps=None,
                                shuffle=False,
                                partition=args.partition,
                                columns_to_drop=columns_to_drop)

        test_data_gen = BatchGen(reader=test_reader,
                                    discretizer=discretizer,
                                    normalizer=normalizer,
                                    batch_size=config["batch_size"],
                                    steps=None,
                                    shuffle=False,
                                    partition=args.partition,
                                    columns_to_drop=columns_to_drop)
        print(f"Number of Steps in Train Data: {train_data_gen.steps}")
        print(f"Number of Steps in Validation Data: {val_data_gen.steps}")
        print(f"Number of Steps in Test Data: {test_data_gen.steps}")

        best_val_loss = float("inf")
        for epoch in range(config["num_epochs"]):
            model.train()
            total_loss = 0
            
            for i in range(train_data_gen.steps):
                batch = next(train_data_gen)
                x, y = batch
                x = torch.FloatTensor(x).to(device)
                y = torch.FloatTensor(y).to(device)
                
                outputs = model(x)

                loss = criterion(outputs, y)
                
                optimizer.zero_grad()
                loss.backward()
                
                if args.model == "transformer":
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                
                total_loss += loss.item()
        
            avg_loss = total_loss / train_data_gen.steps

            model.eval()
            with torch.no_grad():
                val_loss = 0
                for i in range(val_data_gen.steps):
                    batch = next(val_data_gen)
                    x, y = batch
                    x = torch.FloatTensor(x).to(device)
                    y = torch.FloatTensor(y).to(device)
                    
                    outputs = model(x)

                    loss = criterion(outputs, y)
                    
                    val_loss += loss.item()
                
                avg_val_loss = val_loss / val_data_gen.steps

            print(f'Epoch [{epoch+1}/{config["num_epochs"]}], Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
            
            wandb.log({
                "train_loss": avg_loss,
                "val_loss": avg_val_loss
            })

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), args.model_name)


        wandb.log_artifact(args.model_name, name="deterministic_model", type="model")
        model.load_state_dict(torch.load(args.model_name))

        model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for i in range(test_data_gen.steps):
                batch = next(test_data_gen)
                x, y = batch
                x = torch.FloatTensor(x).to(device)
                y = torch.FloatTensor(y).to(device)
                
                outputs = model(x)
        
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(y.cpu().numpy())

        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        mse = np.mean((all_predictions - all_targets) ** 2)

        rmse = np.sqrt(mse)

        print(f'Final Test MSE: {mse:.4f}')
        print(f'Final Test RMSE: {rmse:.4f}') 

        wandb.log({
            "MSE": mse,
            "RMSE": rmse,
            "Dataset Fraction": args.dataset_fraction
        })

        return rmse

    finally:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    utils.add_common_arguments(parser)

    parser.add_argument('--deep_supervision', dest='deep_supervision', action='store_true')

    parser.set_defaults(deep_supervision=False)

    parser.add_argument('--partition', type=str, default='custom',
                        help="log, custom, none")

    parser.add_argument('--data', type=str, help='Path to the data of length-of-stay task',
                        default=os.path.join(os.path.dirname(__file__), '../../data/length-of-stay/'))

    parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                        default='.')

    parser.add_argument('--num_train_samples', type=int, default=None, help='Number of training samples to use')

    parser.add_argument('--num_mc_samples', type=int, default=25, help='Number of Monte Carlo samples for uncertainty estimation')

    parser.add_argument("--model", type=str, default="lstm", help="lstm, rnn, gru, transformer")

    parser.add_argument("--dataset_fraction", type=float, default=1.0, help="fraction of data that is being kept. 0.5 is half the data")

    parser.add_argument("--model_name", type=str, help="Name for the model file")
    
    parser.add_argument("--num_trials", type=int, help="Amount of Optuna optimizer trials")

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

    print('  Value (RMSE): ', trial.value)
    print('  Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')

    with open(os.path.join(args.output_dir, 'best_hyperparams.txt'), 'w') as f:
        f.write(f"Best RMSE: {trial.value}\n")
        for key, value in trial.params.items():
            f.write(f"{key}: {value}\n")

    optuna.visualization.plot_optimization_history(study)
    plt.savefig(os.path.join(args.output_dir, 'optimization_history.png'))
    optuna.visualization.plot_param_importances(study)
    plt.savefig(os.path.join(args.output_dir, 'param_importances.png'))

if __name__ == "__main__":
    main()
