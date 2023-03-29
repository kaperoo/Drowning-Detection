# Purpose: CNN model for keypoint classification
# classifies 17x3 keypoint data into 4 classes: drown, swim, misc, idle
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from read import KeypointDataset
from torch import nn, save, load

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(17, 163, kernel_size=(3, 3), padding=(1, 1)) # prev kernel_size=(1, 3), padding=(0, 1)
        self.conv2 = nn.Conv2d(163, 162, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(162, 235, kernel_size=(3, 3), padding=(1, 1))
        # self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(235 * 3, 256)
        self.fc2 = nn.Linear(256, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 17, 3, 1) # reshape to (batch_size, channels, height, width)
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        # x = self.pool(self.relu(self.conv4(x)))
        x = x.view(-1, 235 * 3)
        # print(x.shape)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
model = CNNModel()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()

# learning_rate = 1e-3
learning_rate = 0.00024149705692512547

# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if __name__ == "__main__":

    train_dataset = KeypointDataset("keypoints_norm17x3.pt", "labels_norm.pt")

    #data size is 49320
    batch_size = int(90/3)
    n_iters = int(1096*43*3)
    num_epochs = n_iters / (len(train_dataset) / batch_size)
    num_epochs = int(num_epochs)

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
            images = images.unsqueeze(1)
            # print(images.shape)

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            outputs = model(images)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels.long())


            # Getting gradients w.r.t. parameters
            loss.backward()

            runnung_loss += loss.item()

            # Updating parameters
            optimizer.step()

            if i % 100 == 9:
                print("Epoch: ", epoch, "Loss: ", runnung_loss/10)
                runnung_loss = 0.0
                

        # print(f"Epoch: ", epoch, "Loss: ", loss.item())

    with open('model_state.pt', 'wb') as f: 
        save(model.state_dict(), f) 