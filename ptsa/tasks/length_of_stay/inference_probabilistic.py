import os
import logging
import argparse
from typing import Dict, List

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn

import wandb
import optuna
import random

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, accuracy_score, precision_score, recall_score, roc_auc_score, f1_score

from ptsa.tasks.readers import LengthOfStayReader
from ptsa.utils.preprocessing import Discretizer, Normalizer
from ptsa.tasks.in_hospital_mortality.utils import load_data
from ptsa.utils import utils

from ptsa.tasks.length_of_stay.utils import BatchGen

logging.basicConfig(level=logging.INFO)

class LOSProbabilisticInference:
    def __init__(self, config: Dict, 
                 data_path: str, 
                 model_path: str, 
                 model_name: str, 
                 device: str, 
                 num_batches_inference: int,
                 limit_num_test_sampled: bool,
                 probabilistic: bool) -> None:
        self.logger = logging.getLogger(__name__)
        self.device = device
        self.config = config 
        self.data_path = data_path
        self.model_path = model_path
        self.model_name = model_name
        self.num_batches_inference = num_batches_inference * 64
        self.limit_num_test_sampled = limit_num_test_sampled
        self.probabilistic = probabilistic

    def _load_model_deterministic(self):
        from ptsa.models.deterministic.lstm import LSTM
        from ptsa.models.deterministic.rnn import RNN
        from ptsa.models.deterministic.gru import GRU
        from ptsa.models.deterministic.transformer import TransformerLOS
        
        print(self.config)

        if self.model_name == "LSTM":
            model = LSTM(self.config["input_size"], self.config["hidden_size"], self.config["num_layers"], self.config["dropout"]).to(self.device)
        elif self.model_name == "RNN":
            model = RNN(self.config["input_size"], self.config["hidden_size"], self.config["num_layers"], self.config["dropout"]).to(self.device)
        elif self.model_name == "GRU":
            model = GRU(self.config["input_size"], self.config["hidden_size"], self.config["num_layers"], self.config["dropout"]).to(self.device)
        elif self.model_name == "transformer":
            model = TransformerLOS(input_size=self.config["input_size"],
                                    d_model=self.config["d_model"],
                                    nhead=self.config["nhead"],
                                    num_layers=self.config["num_layers"],
                                    dropout=self.config["dropout"],
                                    dim_feedforward=self.config["dim_feedforward"]).to(self.device)
        model.load_state_dict(torch.load(self.model_path, weights_only=True))

        return model



    def _load_model_probabilisitc(self):
        from ptsa.models.probabilistic.bayesian_lstm import LSTM 
        from ptsa.models.probabilistic.rnn import RNN
        from ptsa.models.probabilistic.gru import GRU
        from ptsa.models.probabilistic.transformer import TransformerLOS
        
        if self.model_name == "LSTM":
            model = LSTM(self.config["input_size"], self.config["hidden_size"], self.config["num_layers"], self.config["dropout"]).to(self.device)
        elif self.model_name == "RNN":
            model = RNN(self.config["input_size"], self.config["hidden_size"], self.config["num_layers"], self.config["dropout"]).to(self.device)
        elif self.model_name == "GRU":
            model = GRU(self.config["input_size"], self.config["hidden_size"], self.config["num_layers"], self.config["dropout"]).to(self.device)
        elif self.model_name == "transformer":
            model = TransformerLOS(input_size=self.config["input_size"],
                                    d_model=self.config["d_model"],
                                    nhead=self.config["nhead"],
                                    num_layers=self.config["num_layers"],
                                    dropout=self.config["dropout"],
                                    dim_feedforward=self.config["dim_feedforward"]).to(self.device)
        model.load_state_dict(torch.load(self.model_path, weights_only=True))

        return model

    def even_out_number_of_data_points(self, data):
        data_points, labels = data[0], data[1]
        self.logger.info("Number of Samples: %s", len(labels))
        positive_indices = [i for i, label in enumerate(labels) if label == 1]
        negative_indices = [i for i, label in enumerate(labels) if label == 0]
        
        self.logger.info("Number of Positive Samples: %s", len(positive_indices))
        self.logger.info("Number of Negative Samples: %s", len(negative_indices))
        target_size = min(len(positive_indices), len(negative_indices))

        sampled_positive_indices = random.sample(positive_indices, target_size)
        sampled_negative_indices = random.sample(negative_indices, target_size)

        balanced_indices = sampled_positive_indices + sampled_negative_indices
        random.shuffle(balanced_indices)

        balanced_data_points = [data_points[i] for i in balanced_indices]
        balanced_labels = [labels[i] for i in balanced_indices]
        self.logger.info("Number of Balanced Samples: %s", len(balanced_labels))
        
        return (balanced_data_points, balanced_labels)

    
    def load_test_data(self):
        all_data = LengthOfStayReader(dataset_dir=os.path.join(self.data_path, 'train'),
                                        listfile=os.path.join(self.data_path, 'train/listfile.csv'))

        train_data, val_data = train_test_split(all_data._data, test_size=0.2, random_state=42)

        train_reader = LengthOfStayReader(dataset_dir=os.path.join(self.data_path, 'train'))
        train_reader._data = train_data

        if self.limit_num_test_sampled:
            if self.num_batches_inference > len(train_reader._data):
                raise ValueError(f"Requested amount of data is too high. Try lower num of batches")
            
            max_start_idx = len(train_reader._data) - self.num_batches_inference
            start_idx = np.random.randint(0, max_start_idx + 1)
            
            train_reader._data = train_reader._data[start_idx:start_idx + self.num_batches_inference]
        
        val_reader = LengthOfStayReader(dataset_dir=os.path.join(self.data_path, 'train'))
        val_reader._data = val_data


        test_reader = LengthOfStayReader(dataset_dir=os.path.join(self.data_path, "test"),
                                        listfile=os.path.join(self.data_path, "test/listfile.csv"))
        if self.limit_num_test_sampled:
            if self.num_batches_inference > len(test_reader._data):
                raise ValueError(f"Requested amount of data is too high. Try lower num of batches")
            
            max_start_idx = len(test_reader._data) - self.num_batches_inference
            start_idx = np.random.randint(0, max_start_idx + 1)
            
            test_reader._data = test_reader._data[start_idx:start_idx + self.num_batches_inference]
        # test_reader = LengthOfStayReader(dataset_dir=os.path.join(args.data, 'train'))
        # test_reader._data = test_data

        discretizer = Discretizer(timestep=1.0,
                                    store_masks=True,
                                    impute_strategy='previous',
                                    start_time='zero')

        discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
        cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

        normalizer = Normalizer(fields=cont_channels)
        normalizer_state = None

        if normalizer_state is None:
            normalizer_state = 'los_ts{}.input_str_previous.start_time_zero.n5e4.normalizer'.format("1.0")
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
                                    batch_size=self.config["batch_size"],
                                    steps=None,
                                    shuffle=True,
                                    partition="custom",
                                    columns_to_drop=columns_to_drop)

        val_data_gen = BatchGen(reader=val_reader,
                                discretizer=discretizer,
                                normalizer=normalizer,
                                batch_size=self.config["batch_size"],
                                steps=None,
                                shuffle=False,
                                partition="custom",
                                columns_to_drop=columns_to_drop)

        test_data_gen = BatchGen(reader=test_reader,
                                    discretizer=discretizer,
                                    normalizer=normalizer,
                                    batch_size=self.config["batch_size"],
                                    steps=None,
                                    shuffle=False,
                                    partition="custom",
                                    columns_to_drop=columns_to_drop)


        return train_data_gen, val_data_gen, test_data_gen

    def infer_on_data_points(self, test_data):
        if self.probabilistic:
            model = self._load_model_probabilisitc()
        else:
            model = self._load_model_deterministic()

        model.eval()
        all_predictions = []
        all_targets = []
        all_uncertainties = []

        with torch.no_grad():
            for i in range(test_data.steps):
                batch = next(test_data)
                x, y = batch
                x = torch.FloatTensor(x).to(self.device)
                y = torch.FloatTensor(y).to(self.device)
                
                if self.probabilistic:
                    mean, variance = model.predict_with_uncertainty(x, num_samples=self.config["num_mc_samples"])
                    
                    all_predictions.append(mean.cpu().numpy())
                    all_uncertainties.append(variance.cpu().numpy())
                    all_targets.append(y.cpu().numpy())
                else:
                    outputs = model(x)
                    all_predictions.append(outputs.cpu().numpy())
                    all_targets.append(y.cpu().numpy())

        return all_predictions, all_uncertainties, all_targets 
