import torch
import torch.nn as nn
import torchvision.transforms as transforms
from read import KeypointDataset
from torch import nn, save, load

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(17, 32, kernel_size=(1, 3), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1, 3), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(1, 3), padding=(0, 1))
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))

        self.rnn = nn.GRU(128, 128, num_layers=2, batch_first=True, dropout=0.5, bidirectional=True)

        self.fc = nn.Linear(128 * 2, 4)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Convolutional layers
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))

        # Reshape for RNN
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, height * width).transpose(1, 2)

        # Recurrent layer
        x, _ = self.rnn(x)

        # Classification layer
        x = self.fc(x[:, -1, :])
        return x
    
model = CNNModel()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()

learning_rate = 1e-3

# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if __name__ == "__main__":

    train_dataset = KeypointDataset("keypoints_norm17x3.pt", "labels_norm.pt")

    #data size is 49320 * 2
    batch_size = int(90)
    n_iters = int(1096*100)
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
            images = images.unsqueeze(-1)

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