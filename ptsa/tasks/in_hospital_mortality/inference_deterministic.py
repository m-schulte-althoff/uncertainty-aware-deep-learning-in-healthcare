import os
import logging
import argparse
from typing import Dict

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

from ptsa.tasks.in_hospital_mortality.train_deterministic import remove_columns

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

class IHMModelInference:
    def __init__(self, config: Dict, 
                    data_path: str, 
                    model_path: str, 
                    model_name: str, 
                    device: str,
                    probabilistic: bool
                 ) -> None:
            self.logger = logging.getLogger(__name__)
            self.device = device
            self.config = config 
            self.data_path = data_path
            self.model_path = model_path
            self.model_name = model_name
            self.probabilistic = probabilistic

    def _load_deterministic_model(self):
        from ptsa.models.deterministic.lstm_classification import LSTM 
        from ptsa.models.deterministic.rnn_classification import RNN
        from ptsa.models.deterministic.gru_classification import GRU
            
        if self.model_name == "LSTM":
            model = LSTM(self.config["input_size"], self.config["hidden_size"], self.config["num_layers"], self.config["dropout"]).to(self.device)
        elif self.model_name == "RNN":
            model = RNN(self.config["input_size"], self.config["hidden_size"], self.config["num_layers"], self.config["dropout"]).to(self.device)
        elif self.model_name == "GRU":
            model = GRU(self.config["input_size"], self.config["hidden_size"], self.config["num_layers"], self.config["dropout"]).to(self.device)
        elif self.model_name == "transformer":
            model = TransformerIHM(input_size=self.config["input_size"],
                                            d_model=self.config["d_model"],
                                            nhead=self.config["nhead"],
                                            num_layers=self.config["num_layers"],
                                            dropout=self.config["dropout"],
                                            dim_feedforward=self.config["dim_feedforward"]).to(self.device)


        model.load_state_dict(torch.load(self.model_path, weights_only=True))

        return model

    def _load_probabilistic_model(self):
        from ptsa.models.probabilistic.lstm_classification import LSTM 
        from ptsa.models.probabilistic.rnn_classification import RNN
        from ptsa.models.probabilistic.gru_classification import GRU
        from ptsa.models.probabilistic.transformer_classification import TransformerIHM

        if self.model_name == "LSTM":
            model = LSTM(self.config["input_size"], self.config["hidden_size"], self.config["num_layers"], self.config["dropout"]).to(self.device)
        elif self.model_name == "RNN":
            model = RNN(self.config["input_size"], self.config["hidden_size"], self.config["num_layers"], self.config["dropout"]).to(self.device)
        elif self.model_name == "GRU":
            model = GRU(self.config["input_size"], self.config["hidden_size"], self.config["num_layers"], self.config["dropout"]).to(self.device)
        elif self.model_name == "transformer":
            model = TransformerIHM(input_size=self.config["input_size"],
                                            d_model=self.config["d_model"],
                                            nhead=self.config["nhead"],
                                            num_layers=self.config["num_layers"],
                                            dropout=self.config["dropout"],
                                            dim_feedforward=self.config["dim_feedforward"]).to(self.device)


        model.load_state_dict(torch.load(self.model_path, weights_only=True))

        return model


    def _even_out_number_of_data_points(self, data):
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
    
    def load_test_data(self):
        all_reader = InHospitalMortalityReader(
            dataset_dir=os.path.join(self.data_path, 'train'), 
            listfile=os.path.join(self.data_path, 'train/listfile.csv'), 
            period_length=48.0
        )

        train_data, val_data = train_test_split(all_reader._data, test_size=0.2, random_state=43)

        train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(self.data_path, 'train'), listfile=os.path.join(self.data_path, 'train/listfile.csv'), period_length=48.0)
        train_reader._data = train_data

        val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(self.data_path, 'train'), listfile=os.path.join(self.data_path, 'train/listfile.csv'), period_length=48.0)
        val_reader._data = val_data

        test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(self.data_path, 'test'), listfile=os.path.join(self.data_path, 'test/listfile.csv'), period_length=48.0)

        discretizer = Discretizer(
            timestep=float(1.0), 
            store_masks=True, 
            impute_strategy='previous',
            start_time='zero'
        )
        discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
        cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

        normalizer = Normalizer(fields=cont_channels)
        normalizer_state = None
        if normalizer_state is None:
            normalizer_state = f'ihm_ts1.0.input_str_previous.start_time_zero.normalizer'
            normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
        normalizer.load_params(normalizer_state)
        
        columns_to_remove = [
            "Glascow coma scale motor response", 
            "Capillary refill rate", 
            "Glascow coma scale verbal response",
            "Glascow coma scale eye opening"
        ]

        train_raw_data = load_data(train_reader, discretizer, normalizer, False)
        val_raw_data = load_data(val_reader, discretizer, normalizer, False)
        test_raw_data = load_data(test_reader, discretizer, normalizer, False)

        train_raw_data = remove_columns(train_raw_data, discretizer_header, columns_to_remove)
        val_raw_data = remove_columns(val_raw_data, discretizer_header, columns_to_remove)
        test_raw_data = remove_columns(test_raw_data, discretizer_header, columns_to_remove)
        
        train_raw = self._even_out_number_of_data_points(train_raw_data)
        val_raw = self._even_out_number_of_data_points(val_raw_data)
        test_raw = self._even_out_number_of_data_points(test_raw_data)

        return train_raw, val_raw, test_raw

    def infer_on_data_points(self, test_data):
        if self.probabilistic:
            model = self._load_probabilistic_model()
        else:
            model = self._load_deterministic_model()

        model.eval()
        all_predictions = []
        all_targets = []
        all_uncertainties = []

        with torch.no_grad():
            for i in range(len(test_data[0])):
                x, y = test_data[0][i], test_data[1]
                x = torch.FloatTensor(x).to(self.device)
                y = torch.FloatTensor([y[i] if isinstance(y, (list, np.ndarray)) else y]).to(self.device)

                if x.dim() == 2:
                    x = x.unsqueeze(0)
                
                if self.probabilistic:
                    mean, variance = model.predict_with_uncertainty(x, num_samples=self.config["num_mc_samples"])
                
                    all_predictions.append(mean.cpu().numpy())
                    all_uncertainties.append(variance.cpu().numpy())
                    all_targets.append(y.cpu().numpy())

                else:
                    outputs = model(x).view(-1)
                    all_predictions.append(outputs.cpu().numpy())
                    all_targets.append(y.cpu().numpy())


        return all_predictions, all_targets, all_uncertainties 
