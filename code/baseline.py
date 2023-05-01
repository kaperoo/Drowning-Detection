import torch
import torch.nn as nn
import torch.optim as optim
from read import KeypointDataset

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(51, 50)
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(50, 50)
        self.layer4 = nn.Linear(50, 3)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.activation(self.layer3(x))
        x = self.layer4(x)
        return x

# Instantiate the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = NeuralNetwork().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

if __name__ == "__main__":
    # Load the dataset
    train_dataset = KeypointDataset("keypoints_norm17x3_no_misc.pt", "labels_norm_no_misc.pt")

    # Training loop
    num_epochs = 100
    batch_size = 30

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=batch_size,
                                                shuffle=True)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (labels,images) in enumerate(train_loader):
            
            images = images.requires_grad_().to(device)
            labels = labels.to(device)

            images = images.view(-1, 51)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.long())
            loss.backward()

            running_loss += loss.item()

            optimizer.step()

            if i % 100 == 9:
                print("Epoch: ", epoch, "Loss: ", running_loss/10)
                running_loss = 0.0

    with open('model_baseline.pt', 'wb') as f: 
        torch.save(model.state_dict(), f) 