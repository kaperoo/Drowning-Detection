# Purpose: CNN model for keypoint classification
# classifies 17x3 keypoint data into 4 classes: drown, swim, misc, idle
import torch
import torch.nn as nn
from readrnntest import KeypointDataset
from torch import nn, save


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.hidden_size = 114
        self.num_layers = 4
        self.cnn = nn.Sequential(
            nn.Conv2d(17, 187, kernel_size=(3, 3), padding=(1, 1)), # 17, 86, 218, 227 / 17, 187, 176, 66
            nn.ReLU(),
            nn.Conv2d(187, 176, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(176, 66, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
        )
        # Add an GRU layer
        self.gru = nn.GRU(input_size=66 * 3, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)

        # Modify the fully connected layers
        self.fc1 = nn.Linear(self.hidden_size, 256)
        self.fc2 = nn.Linear(256, 3)


    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()

        ii = 0
        y = self.cnn(x[:,ii])
        y = y.view(batch_size, -1)
        output, hidden = self.gru(y.unsqueeze(1))
        # GRU forward pass
        for ii in range(1,seq_len):
            y = self.cnn(x[:,ii])
            y = y.view(batch_size, -1)
            out, hidden = self.gru(y.unsqueeze(1),hidden)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

model = CNNModel()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()

learning_rate = 3.564548037116001e-05#9.143381850322516e-05 # 3.564548037116001e-05

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if __name__ == "__main__":
    train_dataset = KeypointDataset("C:\\Users\\User\\Desktop\\Code\\FYP\\keypoints_30")


    batch_size = 16
    num_epochs = 128

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=batch_size,
                                                shuffle=True)

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

            if i % 100 == 9:
                print("Epoch: ", epoch, "Loss: ", runnung_loss/10)
                runnung_loss = 0.0
                

        # print(f"Epoch: ", epoch, "Loss: ", loss.item())

    with open('model_cnnrnn.pt', 'wb') as f: 
        save(model.state_dict(), f) 