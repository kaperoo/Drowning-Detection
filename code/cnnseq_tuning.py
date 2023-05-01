# PURPOSE: Find oprimal hyperparameters for CNN model using Optuna
# this is for model based on 17 keypoints with x,y and conf
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from readrnn import KeypointDataset
from torch import nn, save, load

import optuna
import numpy as np

class CNNModel(nn.Module):
    def __init__(self, conv1_channels, conv2_channels, conv3_channels):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(17, conv1_channels, kernel_size=(3, 3), padding=(1, 1)) # prev kernel_size=(1, 3), padding=(0, 1)
        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(conv2_channels, conv3_channels, kernel_size=(3, 3), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.fc1 = nn.Linear(conv3_channels, 256)
        self.fc2 = nn.Linear(256, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        # Initializing hidden state for first input using method defined below
        x = x.view(batch_size * seq_len, c, h, w)
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(batch_size, seq_len, -1)

        # Use the last output of the LSTM
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# train_dataset = KeypointDataset("keypoints_norm17x3_no_misc.pt", "labels_norm_no_misc.pt", seq_length=seq_len)


def objective(trial):
    # Define hyperparameters to optimize
    lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    num_epochs = trial.suggest_int('num_epochs', 20, 50)
    conv1_channels = trial.suggest_int('conv1_channels', 50, 256)
    conv2_channels = trial.suggest_int('conv2_channels', 50, 256)
    conv3_channels = trial.suggest_int('conv3_channels', 50, 256)
    seq_len = trial.suggest_categorical('seq_len', [5, 10, 15, 30])

    train_dataset = KeypointDataset("keypoints_norm17x3_no_misc.pt", "labels_norm_no_misc.pt", seq_length=seq_len)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                batch_size=batch_size,
                                                shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_data,
                                                batch_size=batch_size,
                                                shuffle=True)

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
            images = images.requires_grad_().to(device)
            labels = labels.to(device)

            images = images.unsqueeze(-1)
            # print(images.shape)
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            outputs = model(images)

            # Calculate Loss: softmax --> cross entropy loss
            seq_losses = [criterion(outputs[:, seq_idx, :], labels[:, seq_idx].long()) for seq_idx in range(seq_len)]
            loss = sum(seq_losses) / seq_len

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
                images = images.unsqueeze(-1)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 2)  # Get the predicted labels for each time step
                for seq_idx in range(seq_len):
                    total += labels[:, seq_idx].size(0)
                    total_correct += (predicted[:, seq_idx] == labels[:, seq_idx]).sum().item()
            accuracy = total_correct / total
        trial.report(accuracy, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=128)

best_trial = study.best_trial
best_lr = best_trial.params['learning_rate']
best_batch_size = best_trial.params['batch_size']
best_num_epochs = best_trial.params['num_epochs']
best_conv1_channels = best_trial.params['conv1_channels']
best_conv2_channels = best_trial.params['conv2_channels']
best_conv3_channels = best_trial.params['conv3_channels']
best_seq_len = best_trial.params['seq_len']

print('Best trial:')
print('  Value: {}'.format(best_trial.value))
print('  Params: ')
for key, value in best_trial.params.items():
    print('    {}: {}'.format(key, value))