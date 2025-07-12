import os
import argparse

import torch
from torch import nn
from sklearn.model_selection import train_test_split
import wandb
import numpy as np

from ptsa.tasks.readers import LengthOfStayReader
from ptsa.tasks.length_of_stay.utils import BatchGen
from ptsa.utils.preprocessing import Normalizer, Discretizer
from ptsa.tasks.length_of_stay.utils import utils

from ptsa.models.deterministic.lstm import LSTM 
from ptsa.models.deterministic.rnn import RNN
from ptsa.models.deterministic.gru import GRU


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

parser.add_argument("--model", type=str, default="lstm", help="lstm, rnn, gru, transformer")

parser.add_argument("--model_name", type=str, help="Name for the model file")

args = parser.parse_args()

config = {
    "input_size": 76,
    "hidden_size": 64,
    "num_layers": 2,
    "learning_rate": 0.001,
    "num_epochs": 5,
    "batch_size": 64,
    "dropout": 0.2
}

wandb.init(project=f"deterministic_{args.model}_los", config=config)

device = "cuda" if torch.cuda.is_available() else "cpu" 

model = nn.Module()
if args.model == "lstm":
    model = LSTM(config["input_size"], config["hidden_size"], config["num_layers"], config["dropout"]).to(device)
    print(f"Using LSTM on {next(model.parameters()).device}")
elif args.model == "rnn":
    model = RNN(config["input_size"], config["hidden_size"], config["num_layers"], config["dropout"]).to(device)
    print(f"Using RNN on {next(model.parameters()).device}")
elif args.model == "gru":
    model = GRU(config["input_size"], config["hidden_size"], config["num_layers"], config["dropout"]).to(device)    
    print(f"Using RNN on {next(model.parameters()).device}")

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

val_reader = LengthOfStayReader(dataset_dir=os.path.join(args.data, 'train'))
val_reader._data = val_data

test_reader = LengthOfStayReader(dataset_dir=os.path.join(args.data, "test"),
                                 listfile=os.path.join(args.data, "test/listfile.csv"))
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

train_data_gen = BatchGen(reader=train_reader,
                            discretizer=discretizer,
                            normalizer=normalizer,
                            batch_size=config["batch_size"],
                            steps=None,
                            shuffle=True,
                            partition=args.partition)

val_data_gen = BatchGen(reader=val_reader,
                        discretizer=discretizer,
                        normalizer=normalizer,
                        batch_size=config["batch_size"],
                        steps=None,
                        shuffle=False,
                        partition=args.partition)

test_data_gen = BatchGen(reader=test_reader,
                            discretizer=discretizer,
                            normalizer=normalizer,
                            batch_size=config["batch_size"],
                            steps=None,
                            shuffle=False,
                            partition=args.partition)


print(f"Number of Steps in Train Data: {train_data_gen.steps}")
print(f"Number of Steps in Validation Data: {val_data_gen.steps}")

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


wandb.log_artifact(args.model_name, name="rnn_deterministic_model", type="model")
model.load_state_dict(torch.load(args.model_name))

model.eval()
all_predictions = []
all_targets = []

print(f"Number of Steps in Test Data: {test_data_gen.steps}")
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
    "RMSE": rmse
})


wandb.finish()
