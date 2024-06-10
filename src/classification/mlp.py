import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler, TensorDataset

import argparse
from torch.utils.data.dataloader import _collate_fn_t, _worker_init_fn_t
import yaml

import numpy as np

from tqdm import tqdm

class ExpData():
    """
    This is the dataset class for the features extracted from the hooked LLM, for experiment.
    Given the parent folder of the dataset, this class will load the data according to the preset structure.
    The structure of the data is specified as:
    ./features.npy
    ./labels.npy
    """
    def __init__(self, data_path: str):
        self.features = torch.tensor(np.load(f'{data_path}/features.npy'))
        self.labels = torch.tensor(np.load(f'{data_path}/labels.npy'))

class MLP(nn.Module):
    def __init__(self, config, input_size: int):
        super(MLP, self).__init__()
        layers = []
        for layer_config in config['model']['layers']:
            if layer_config['type'] == 'Linear':
                layers.append(nn.Linear(layer_config['input_size'], layer_config['output_size']))
            elif layer_config['type'] == 'ReLU':
                layers.append(nn.ReLU())
            elif layer_config['type'] == 'Sigmoid':
                layers.append(nn.Sigmoid())
            elif layer_config['type'] == 'LayerNorm':
                layers.append(nn.LayerNorm(layer_config['normalized_shape']))
            # Add more layer types as needed
        # add implicit first linear layer
        layers = [nn.Linear(input_size, config['model']['layers'][0]['input_size']), nn.ReLU()] + layers
        # initialize weights
        for layer in layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)
    
class MLPClassifier:
    def __init__(self, config, train_features: torch.Tensor, train_labels: torch.Tensor) -> None:
        self.config = config
        self.train_features = train_features
        self.train_labels = train_labels

        # Normalize the training data
        self.mean = train_features.mean(dim=0)
        self.std = train_features.std(dim=0)
        self.train_features = (train_features - self.mean) / self.std

        # calculate class weights
        class_counts = train_labels.long().bincount()
        self.class_weights = 1 / class_counts.float()
        self.class_weights /= self.class_weights.sum()
        if 'class_weights' in config['training']:
            self.class_weights *= torch.tensor(config['training']['class_weights'])
        print(f'Class weights: {self.class_weights}')

        self.classify_threshold = 0.5

        self.input_size = train_features.shape[1]
        self.model = MLP(config, self.input_size)
        self.train_model()

    def train_model(self):
        # Check for GPU availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        print(f"Training on {device}")

        # Move the model to the appropriate device
        self.model.to(device)

        self.model.train()

        batch_size = self.config['training']['batch_size']
        num_epochs = self.config['training']['epochs']
        learning_rate = self.config['training']['learning_rate']
        weight_decay = self.config['training']['weight_decay']
        dataset = TensorDataset(self.train_features.to(device), self.train_labels.to(device))
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        criterion = nn.BCELoss(reduction='none')

        for epoch_id in tqdm(range(num_epochs), desc="Epochs"):
            for inputs, labels in data_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)  # Outputs are raw logits
                # Apply class weights manually
                loss = criterion(outputs, labels[..., None])
                weights = labels * self.class_weights[1] + (1 - labels) * self.class_weights[0]
                loss = (loss * weights).mean()  # Manually weight the loss and take the mean
                loss.backward()
                optimizer.step()
            # print(f'Epoch {epoch_id+1}/{num_epochs}, Loss: {loss.item()}')

    def evaluate(self, features: torch.Tensor, labels: torch.Tensor):
        features = features.to(self.device)
        # Normalize features using the same mean and std as used for training
        features = (features - self.mean) / self.std

        # Set model to evaluation mode
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(features)
            # Assuming binary classification with a threshold of 0.5
            predictions = (outputs > self.classify_threshold).float()
            accuracy = (predictions == labels[..., None]).float().mean()

        return accuracy.item()

    def score(self, features: torch.Tensor):
        features = features.to(self.device)
        # Normalize features using the same mean and std as used for training
        features = (features - self.mean) / self.std

        # Set model to evaluation mode
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(features)

        return outputs
