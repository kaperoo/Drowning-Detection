# Purpose: CNN model for keypoint classification
# classifies 17x3 keypoint data into 4 classes: drown, swim, misc, idle
import torch
import torch.nn as nn
from readrnn import KeypointDataset
from torch import nn, save

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(17, 86, kernel_size=(3, 3), padding=(1, 1)) # prev kernel_size=(1, 3), padding=(0, 1)
        self.conv2 = nn.Conv2d(86, 218, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(218, 227, kernel_size=(3, 3), padding=(1, 1))
        # self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu = nn.ReLU()

        # Modify the fully connected layers
        self.fc1 = nn.Linear(227*3, 256)
        self.fc2 = nn.Linear(256, 3)


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

model = CNNModel()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()

# learning_rate = 1e-3
learning_rate = 3.564548037116001e-05

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if __name__ == "__main__":
    seq_len = 30
    train_dataset = KeypointDataset("keypoints_norm17x3_no_misc.pt", "labels_norm_no_misc.pt", seq_length=seq_len)

    batch_size = 16
    num_epochs = 45

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=batch_size,
                                                shuffle=True)

    for epoch in range(num_epochs):
        runnung_loss = 0.0
        for i, (labels, images) in enumerate(train_loader):
            # Load images as a torch tensor with gradient accumulation abilities
            images = images.requires_grad_().to(device)
            labels = labels.to(device)

            # make input channel = 1
            # print(images.shape)
            images = images.unsqueeze(-1)
            # print(images.shape)

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            outputs = model(images)
            # Calculate Loss: softmax --> cross entropy loss independently for each sequence element
            seq_losses = [criterion(outputs[:, seq_idx, :], labels[:, seq_idx].long()) for seq_idx in range(seq_len)]
            loss = sum(seq_losses) / seq_len

            # Getting gradients w.r.t. parameters
            loss.backward()

            runnung_loss += loss.item()

            # Updating parameters
            optimizer.step()

            if i % 100 == 9:
                print("Epoch: ", epoch, "Loss: ", runnung_loss/10)
                runnung_loss = 0.0
                
    with open('model_state_seq.pt', 'wb') as f: 
        save(model.state_dict(), f) 