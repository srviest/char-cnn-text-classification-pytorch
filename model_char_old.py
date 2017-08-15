import torch
import torch.nn as nn
import torch.nn.functional as F

# class InferenceBatchLogSoftmax(nn.Module):
#     def forward(self, input_):
#         if not self.training:
#             batch_size = input_.size()[0]
#             return torch.stack([F.log_softmax(input_[i]) for i in range(batch_size)], 0)
#         else:
#             return input_


class  CharCNN(nn.Module):
    
    def __init__(self, num_features):
        super(CharCNN, self).__init__()
        
        self.num_features = num_features
        self.conv1 = nn.Conv1d(self.num_features, 256, kernel_size=7, stride=1, bias=False)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=3)
            
        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=7, stride=1, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, bias=False),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, bias=False),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, bias=False),
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
        self.fc3 = nn.Linear(1024, 4)
        self.softmax = nn.LogSoftmax()
        # self.inference_log_softmax = InferenceBatchLogSoftmax()

    def forward(self, x):
        x = self.conv1(x)
        print('x.size()', x.size())
        x = self.relu1(x)
        print('x.size()', x.size())
        x = self.pool1(x)
        print('x.size()', x.size())
        x = self.conv2(x)
        print('x.size()', x.size())
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        # print('x.size()', x.size())
        # print('x.size(0)', x.size(0))
        x = x.view(x.size(0), -1)
        # print('Collapse x:, ', x.size())
        x = self.fc1(x)
        # print('FC1: ', x.size())
        x = self.fc2(x)
        # print('FC2: ', x.size())
        x = self.fc3(x)
        # print('x: ', x.size())
        # x = self.inference_log_softmax(x)

        x = self.softmax(x)



        return x