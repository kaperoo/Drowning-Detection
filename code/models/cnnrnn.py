# Purpose: CNNRNN model for keypoint classification
# classifies 17x3 keypoint data into 3 classes: drown, swim, idle
import torch
import torch.nn as nn
import sys
from os.path import dirname, join
sys.path.append(join(dirname(__file__), "..\\datasets"))
from datasetrnn import KeypointDataset
from torch import nn, save

# Define CNNRNN model
class CNNRNN(nn.Module):
    def __init__(self):
        super(CNNRNN, self).__init__()
        
        # hyperparameters for GRU layer
        self.hidden_size = 114
        self.num_layers = 4

        # CNN block
        self.cnn = nn.Sequential(
            nn.Conv2d(17, 187, kernel_size=(3, 3), padding=(1, 1)), # 17, 86, 218, 227 / 17, 187, 176, 66
            nn.ReLU(),
            nn.Conv2d(187, 176, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(176, 66, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
        )

        # Recurrent block
        self.gru = nn.GRU(input_size=66 * 3, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)

        # Fully connected layers for output
        self.fc1 = nn.Linear(self.hidden_size, 256)
        self.fc2 = nn.Linear(256, 3)


    def forward(self, x):
        # find the batch size, sequence length, and image size
        batch_size, seq_len, c, h, w = x.size()

        # Loop through sequence
        ii = 0
        y = self.cnn(x[:,ii])
        y = y.view(batch_size, -1)
        output, hidden = self.gru(y.unsqueeze(1))
        # GRU forward pass
        for ii in range(1,seq_len):
            y = self.cnn(x[:,ii])
            y = y.view(batch_size, -1)
            out, hidden = self.gru(y.unsqueeze(1),hidden)
        
        # Process the final hidden state to return the classification result
        out = self.fc1(out)
        out = self.fc2(out)
        return out

model = CNNRNN()

# load the model into the GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
learning_rate = 3.564548037116001e-05
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if __name__ == "__main__":

    # Hyperparameters
    batch_size = 16
    num_epochs = 64

    # Load the dataset
    train_dataset = KeypointDataset("..\\keypoints_30")
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True)
    
    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
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
            # Reshape outputs to be 3D
            outputs = outputs.reshape(-1, 3, 1)

            # Calculate Loss: softmax --> cross entropy loss independently for each sequence element
            loss = criterion(outputs, labels[:,-1].unsqueeze(1).long())

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

            # Print loss every 100 steps
            running_loss += loss.item()
            if i % 100 == 9:
                print("Epoch: ", epoch, "Loss: ", running_loss/10)
                running_loss = 0.0

    # Save the model
    with open('model_cnnrnn.pt', 'wb') as f: 
        save(model.state_dict(), f) 