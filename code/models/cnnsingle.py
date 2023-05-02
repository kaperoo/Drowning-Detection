# Purpose: CNN model for keypoint classizesification
# classifies 17x3 keypoint data into 3 classes: drown, swim, idle
import torch
import torch.nn as nn
import sys
from os.path import dirname, join
sys.path.append(join(dirname(__file__), "..\\datasets"))
from dataset import KeypointDataset
from torch import nn, save

# Define CNNSINGLE model
class CNNSINGLE(nn.Module):
    def __init__(self):
        super(CNNSINGLE, self).__init__()

        # 3 convolutional layers, first with 17 input channels, last with 235 output channels
        self.conv1 = nn.Conv2d(17, 163, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(163, 162, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(162, 235, kernel_size=(3, 3), padding=(1, 1))

        # fully connected layers for 3 class output
        self.fc1 = nn.Linear(235 * 3, 256)
        self.fc2 = nn.Linear(256, 3)

        # activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # reshape the input to be (batch_size, channels, height, width)
        x = x.view(-1, 17, 3, 1)

        # CNN forward pass
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        # flatten the output for each image
        x = x.view(-1, 235 * 3)

        # fully connected layers for output
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# define the model on a GPU if available
model = CNNSINGLE()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
learning_rate = 0.00024149705692512547
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if __name__ == "__main__":

    # load the dataset
    train_dataset = KeypointDataset("keypoints_norm17x3_no_misc.pt", "labels_norm_no_misc.pt")

    # hyperparameters
    batch_size = 30
    num_epochs = 43

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=batch_size,
                                                shuffle=True)

    # train loop
    for epoch in range(num_epochs):
        runnung_loss = 0.0
        for i, (labels, images) in enumerate(train_loader):
            
            # Load images as a torch tensor with gradient accumulation abilities
            images = images.requires_grad_().to(device)
            labels = labels.to(device)

            # add a channel dimension to the images
            images = images.unsqueeze(1)

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

            # print statistics
            runnung_loss += loss.item()
            if i % 100 == 9:
                print("Epoch: ", epoch, "Loss: ", runnung_loss/10)
                runnung_loss = 0.0
                
    # save the model
    with open('model_state.pt', 'wb') as f: 
        save(model.state_dict(), f) 