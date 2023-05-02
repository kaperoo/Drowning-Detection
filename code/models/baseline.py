# A baseline model for the keypoint classification task
import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append("..\\datasets")
from dataset import KeypointDataset

# Defining the model
class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        
        # 51 input features (17 keypoints * 3 coordinates)
        self.layer1 = nn.Linear(51, 50)

        # 50x50x50 hidden layers
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(50, 50)
        self.layer4 = nn.Linear(50, 3)

        # ReLU activation
        self.activation = nn.ReLU()

    def forward(self, x):

        # Passing the input through the network
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.activation(self.layer3(x))
        x = self.layer4(x)
        return x

# Instantiate the model on gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Baseline().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

if __name__ == "__main__":

    # Load the dataset
    train_dataset = KeypointDataset("keypoints_norm17x3_no_misc.pt", 
                                    "labels_norm_no_misc.pt")

    # training parameters
    num_epochs = 100
    batch_size = 30

    # data loader for training
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=batch_size,
                                                shuffle=True)

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (labels,images) in enumerate(train_loader):
            
            # Load images as a torch tensor with gradient accumulation abilities
            images = images.requires_grad_().to(device)
            labels = labels.to(device)

            # Reshape the images to 51 features
            images = images.view(-1, 51)

            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, labels.long())
            
            # Backward pass
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 100 == 9:
                print("Epoch: ", epoch, "Loss: ", running_loss/10)
                running_loss = 0.0

    # Save the model after training
    with open('model_baseline.pt', 'wb') as f: 
        torch.save(model.state_dict(), f) 