# Purpose: CNNSEQ model for keypoint classification
# classifies 17x3 keypoint data into 3 classes: drown, swim, idle
import torch
import torch.nn as nn
import sys
from os.path import dirname, join
sys.path.append(join(dirname(__file__), "..\\datasets"))
from datasetseq import KeypointDataset
from torch import nn, save

# Define CNNSEQ model
class CNNSEQ(nn.Module):
    def __init__(self):
        super(CNNSEQ, self).__init__()

        # 3 convolutional layers with ReLU activations
        self.conv1 = nn.Conv2d(17, 86, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(86, 218, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(218, 227, kernel_size=(3, 3), padding=(1, 1))
        self.relu = nn.ReLU()

        # Fully connected layers for 3 class output
        self.fc1 = nn.Linear(227*3, 256)
        self.fc2 = nn.Linear(256, 3)


    def forward(self, x):
        # find the batch size, sequence length, and image size
        batch_size, seq_len, c, h, w = x.size()

        # reshape the input to be (batch_size * seq_len, c, h, w)
        x = x.view(batch_size * seq_len, c, h, w)

        # forward pass through the convolutional layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        # reshape the output to be (batch_size, seq_len, c*h*w)
        x = x.view(batch_size, seq_len, -1)
        
        # forward pass through the fully connected layers for final output
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# load the model into the GPU if available
model = CNNSEQ()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
learning_rate = 3.564548037116001e-05
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if __name__ == "__main__":
    # Load the dataset
    seq_len = 30
    train_dataset = KeypointDataset("keypoints_norm17x3_no_misc.pt", "labels_norm_no_misc.pt", seq_length=seq_len)

    # Hyperparameters
    batch_size = 16
    num_epochs = 45

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=batch_size,
                                                shuffle=True)

    # Train loop
    for epoch in range(num_epochs):
        runnung_loss = 0.0
        for i, (labels, images) in enumerate(train_loader):
            
            # Load images as a torch tensor with gradient accumulation abilities
            images = images.requires_grad_().to(device)
            labels = labels.to(device)

            # Add a channel dimension to the images
            images = images.unsqueeze(-1)

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            outputs = model(images)
            # Calculate Loss: softmax --> cross entropy loss independently for each sequence element
            seq_losses = [criterion(outputs[:, seq_idx, :], labels[:, seq_idx].long()) for seq_idx in range(seq_len)]
            loss = sum(seq_losses) / seq_len

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

            # Print statistics
            runnung_loss += loss.item()
            if i % 100 == 9:
                print("Epoch: ", epoch, "Loss: ", runnung_loss/10)
                runnung_loss = 0.0
    # Save the model       
    with open('model_cnnseq.pt', 'wb') as f: 
        save(model.state_dict(), f) 