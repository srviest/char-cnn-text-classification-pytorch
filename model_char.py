import torch
import torch.nn as nn
import torch.nn.functional as F

class  CharCNN(nn.Module):
    
    def __init__(self, num_features):
        super(CharCNN, self).__init__()
        
        self.num_features = num_features
        self.conv1 = nn.Sequential(
            nn.Conv1d(self.num_features, 256, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(8704, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(1024, 4),
            nn.LogSoftmax()
        )

    def forward(self, x):
        x = self.conv1(x)
        print('Conv1: ', x.size())
        x = self.conv2(x)
        print('Conv2: ', x.size())
        x = self.conv3(x)
        print('Conv3: ', x.size())
        x = self.conv4(x)
        print('Conv4: ', x.size())
        x = self.conv5(x)
        print('Conv5: ', x.size())
        x = self.conv6(x)
        print('Conv6: ', x.size())
        x = x.view(x.size(0), -1)
        print('Collapse x:, ', x.size())
        x = self.fc1(x)
        print('FC1: ', x.size())
        x = self.fc2(x)
        print('FC2: ', x.size())
        output = self.fc3(x)
        print('output: ', output.size())

        return output