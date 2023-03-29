# PURPOSE: CNN model for video classification
# classifies 120x120 videos into 4 classes: drown, swim, misc, idle
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from video import VideoDataset
from torch import nn, save, load

class VideoCNN(nn.Module):
    def __init__(self):
        super(VideoCNN, self).__init__()
        self.conv1 = nn.Conv3d(3, 50, kernel_size=5)
        self.pool1 = nn.MaxPool3d(kernel_size=2)
        self.conv2 = nn.Conv3d(50, 50, kernel_size=5)
        self.pool2 = nn.MaxPool3d(kernel_size=2)
        self.conv3 = nn.Conv3d(50, 50, kernel_size=5)
        self.fc1 = nn.Linear(50*50*50, 100)
        self.fc2 = nn.Linear(100, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = x.view(-1, 50*50*50)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x
    
model = VideoCNN()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()

learning_rate = 1e-3

# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if __name__ == "__main__":

    train_dataset = VideoDataset("../train_frames")

    batch_size = 1

    video_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=batch_size,
                                                shuffle=True)

    num_epochs = 50

    for epoch in range(num_epochs):
        for i, (videos, labels) in enumerate(video_dataloader):
            

            videos = videos.requires_grad_().to(device)
            labels = labels.to(device)

            # videos = videos.unsqueeze(1)
            # reshape videos to the shape of (batch_size, depth, height, width, channels)
            videos = videos.permute(0, 4, 2, 3, 1)

            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch+1, num_epochs, i+1, len(video_dataloader), loss.item()))
                


    with open('model_state.pt', 'wb') as f: 
        save(model.state_dict(), f) 