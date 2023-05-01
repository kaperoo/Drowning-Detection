# Purpose: CNN model for keypoint classification
# classifies 17x3 keypoint data into 4 classes: drown, swim, misc, idle
import torch
import torch.nn as nn
from readrnntest import KeypointDataset
from torch import nn
import optuna


class CNNModel(nn.Module):
    def __init__(self, conv1_channels, conv2_channels, conv3_channels, hidden_channels, hidden_layers):
        super(CNNModel, self).__init__()
        self.hidden_size = hidden_channels
        self.num_layers = hidden_layers
        self.cnn = nn.Sequential(
            nn.Conv2d(17, conv1_channels, kernel_size=(3, 3), padding=(1, 1)), # prev kernel_size=(1, 3), padding=(0, 1)
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(conv1_channels, conv2_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(conv2_channels, conv3_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            # nn.Linear(227*3, 256),
            # nn.Linear(256, 3)
        )
        # Add an LSTM layer
        self.lstm = nn.GRU(input_size=conv3_channels * 3, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)

        # Modify the fully connected layers
        self.fc1 = nn.Linear(self.hidden_size, 256)
        self.fc2 = nn.Linear(256, 3)


    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()

        # Initializing hidden state for first input using method defined below

        ii = 0
        y = self.cnn(x[:,ii])
        y = y.view(batch_size, -1)
        output, hidden = self.lstm(y.unsqueeze(1))
        # LSTM forward pass
        for ii in range(1,seq_len):
            y = self.cnn(x[:,ii])
            y = y.view(batch_size, -1)
            out, hidden = self.lstm(y.unsqueeze(1),hidden)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    train_dataset = KeypointDataset("C:\\Users\\User\\Desktop\\Code\\FYP\\keypoints_30")
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
    model = CNNModel(conv1_channels, conv2_channels, conv3_channels, hidden_channels, hidden_layers)
    model.to(device)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        runnung_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            # Load images as a torch tensor with gradient accumulation abilities
            images = images.requires_grad_().to(device)
            labels = labels.to(device)
            # make input channel = 1
            # print(images.shape)
            images = images.unsqueeze(-1)
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            outputs = model(images)
            outputs = outputs.reshape(-1, 3, 1)
            # Calculate Loss: softmax --> cross entropy loss independently for each sequence element
            loss = criterion(outputs, labels[:,-1].unsqueeze(1).long())



            # Getting gradients w.r.t. parameters
            loss.backward()

            runnung_loss += loss.item()

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

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

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