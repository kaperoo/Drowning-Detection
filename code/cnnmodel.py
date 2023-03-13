import torch
import torch.nn as nn
import torchvision.transforms as transforms
from read import KeypointDataset
from torch import nn, save, load


train_dataset = KeypointDataset("keypoints.pt", "labels.pt")

# test_dataset = datasets.MNIST(root='./data',
                                    # train = False,
                                    # transform=transforms.ToTensor())

batch_size = int(90/30)
n_iters = int(2740*30)
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)

# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                             batch_size=batch_size,
#                                             shuffle=False)

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5)
        self.fc1 = nn.Linear(128, 4)

    def forward(self, x):
        # input shape: (batch_size, 1, 51)
        x = self.conv1(x)  # output shape: (batch_size, 16, 47)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool1d(x, kernel_size=2, stride=2) # output shape: (batch_size, 16, 23)
        x = self.conv2(x)  # output shape: (batch_size, 32, 19)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool1d(x, kernel_size=2, stride=2) # output shape: (batch_size, 32, 9)
        x = self.conv3(x)  # output shape: (batch_size, 64, 5)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool1d(x, kernel_size=2, stride=2) # output shape: (batch_size, 64, 2)
        x = x.view(-1, 64*2)  # flatten
        x = self.fc1(x)  # output shape: (batch_size, 128)

        return x
    
model = CNNModel()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()

learning_rate = 1e-3

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# for epoch in range(num_epochs):
#     for i, (labels, images) in enumerate(train_loader):
#         # Load images as a torch tensor with gradient accumulation abilities
#         images = images.requires_grad_().to(device)
#         labels = labels.to(device)

#         # make input channel = 1
#         images = images.unsqueeze(1)

#         # Clear gradients w.r.t. parameters
#         optimizer.zero_grad()

#         # Forward pass to get output/logits
#         outputs = model(images)

#         predictions = torch.argmax(outputs, dim=1).float()
#         predictions = predictions.requires_grad_().to(device)

#         # Calculate Loss: softmax --> cross entropy loss
#         loss = criterion(predictions, labels)

#         # Getting gradients w.r.t. parameters
#         loss.backward()

#         # Updating parameters
#         optimizer.step()

#     print(f"Epoch: ", epoch, "Loss: ", loss.item())

# with open('model_state.pt', 'wb') as f: 
#     save(model.state_dict(), f) 