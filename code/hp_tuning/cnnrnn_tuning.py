# Purpose: Tune hyperparameters for CNNRNN model
import torch
import torch.nn as nn
import sys
from os.path import dirname, join
sys.path.append(join(dirname(__file__), "..\\datasets"))
from datasetrnn import KeypointDataset
from torch import nn
import optuna

# Define CNNRNN model
class CNNRNN(nn.Module):
    def __init__(self, conv1_channels, conv2_channels, 
                    conv3_channels, hidden_channels, hidden_layers):
        
        super(CNNRNN, self).__init__()
        
        # hyperparameters for GRU layer
        self.hidden_size = hidden_channels
        self.num_layers = hidden_layers

        # CNN block
        self.cnn = nn.Sequential(
            # 17 input channels (17 keypoints), 3 CNN layers with ReLU activation
            nn.Conv2d(17, conv1_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(conv1_channels, conv2_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(conv2_channels, conv3_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
        )

        # Recurrent block
        self.gru = nn.GRU(input_size=conv3_channels * 3, hidden_size=self.hidden_size, 
                            num_layers=self.num_layers, batch_first=True)

        # Fully connected layers for classification
        self.fc1 = nn.Linear(self.hidden_size, 256)
        self.fc2 = nn.Linear(256, 3)


    def forward(self, x):

        # find the batch size, sequence length, and image size
        batch_size, seq_len, c, h, w = x.size()

        # Loop through sequence
        ii = 0
        y = self.cnn(x[:,ii])
        y = y.view(batch_size, -1)
        output, hidden = self.lstm(y.unsqueeze(1))
        # LSTM forward pass
        for ii in range(1,seq_len):
            y = self.cnn(x[:,ii])
            y = y.view(batch_size, -1)
            out, hidden = self.lstm(y.unsqueeze(1),hidden)

        # Process the final hidden state to return the classification result
        out = self.fc1(out)
        out = self.fc2(out)
        return out

# get the available device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the objective function to optimize hyperparameters
def objective(trial):
    # Define hyperparameters to optimize
    lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
    num_epochs = trial.suggest_int('num_epochs', 20, 64)
    conv1_channels = trial.suggest_int('conv1_channels', 50, 256)
    conv2_channels = trial.suggest_int('conv2_channels', 50, 256)
    conv3_channels = trial.suggest_int('conv3_channels', 50, 256)
    hidden_channels = trial.suggest_int('hidden_channels', 50, 128)
    hidden_layers = trial.suggest_int('hidden_layers', 2, 4)

    # Define training and validation data loaders
    train_dataset = KeypointDataset("..\\keypoints_30")
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
    model = CNNRNN(conv1_channels, conv2_channels, conv3_channels, hidden_channels, hidden_layers)
    model.to(device)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Train loop
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):

            # Load images as a torch tensor with gradient accumulation abilities
            images = images.requires_grad_().to(device)
            labels = labels.to(device)
            
            # Add a channel dimension to the images
            images = images.unsqueeze(-1)

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            outputs = model(images)
            
            # Reshape outputs and labels to match the loss function
            outputs = outputs.reshape(-1, 3, 1)

            # Calculate Loss: softmax --> cross entropy loss independently for each sequence element
            loss = criterion(outputs, labels[:,-1].unsqueeze(1).long())

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

        # Evaluate model on validation set
        with torch.no_grad():
            total_correct = 0
            total = 0
            for images, labels in val_loader:
                images = images.to(device)
                images = images.unsqueeze(-1)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 2)
                total += labels.size(0)
                total_correct += (predicted == labels).sum().item()
            accuracy = total_correct / total
        trial.report(accuracy, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return accuracy

# Create a study and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Print optimization results
best_trial = study.best_trial
best_lr = best_trial.params['learning_rate']
best_batch_size = best_trial.params['batch_size']
best_num_epochs = best_trial.params['num_epochs']
best_conv1_channels = best_trial.params['conv1_channels']
best_conv2_channels = best_trial.params['conv2_channels']
best_conv3_channels = best_trial.params['conv3_channels']
best_hidden_channels = best_trial.params['hidden_channels']
best_hidden_layers = best_trial.params['hidden_layers']

print('Best trial:')
print('  Value: {}'.format(best_trial.value))
print('  Params: ')
for key, value in best_trial.params.items():
    print('    {}: {}'.format(key, value))