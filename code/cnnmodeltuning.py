import torch
import torch.nn as nn
import torchvision.transforms as transforms
from read import KeypointDataset
from torch import nn, save, load
from torch.utils.data import random_split

import optuna

import numpy as np

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(17, 128, kernel_size=(3, 3), padding=(1, 1)) # prev kernel_size=(1, 3), padding=(0, 1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1))
        # self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(128 * 3, 256)
        self.fc2 = nn.Linear(256, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 17, 3, 1) # reshape to (batch_size, channels, height, width)
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        # x = self.pool(self.relu(self.conv4(x)))
        x = x.view(-1, 128 * 3)
        # print(x.shape)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

def train_cnn(config, checkpoint_dir=None, data_dir=None):

    train_dataset = KeypointDataset("keypoints_norm17x3.pt", "labels_norm.pt")
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch"], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config["batch"], shuffle=True)

import optuna
import numpy as np

def objective(trial):
    # Define hyperparameters to optimize
    lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    num_epochs = trial.suggest_int('num_epochs', 10, 50)
    conv1_channels = trial.suggest_int('conv1_channels', 64, 256)
    conv2_channels = trial.suggest_int('conv2_channels', 64, 256)
    conv3_channels = trial.suggest_int('conv3_channels', 64, 256)

    # Define model architecture
    model = CNNModel(conv1_channels, conv2_channels, conv3_channels)
    model.to(device)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Train model and calculate validation accuracy
    for epoch in range(num_epochs):
        for i, (labels, images) in enumerate(train_loader):
            # Load images and labels to device
            images = images.to(device)
            labels = labels.to(device)

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            outputs = model(images)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels.long())

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

        # Evaluate model on validation set
        with torch.no_grad():
            total_correct = 0
            total = 0
            for labels, images in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                total_correct += (predicted == labels).sum().item()
            accuracy = total_correct / total
        trial.report(accuracy, epoch)
        if trial.should_prune(epoch):
            raise optuna.exceptions.TrialPruned()
    return accuracy