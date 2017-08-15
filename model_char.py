import torch
import torch.nn as nn
import torch.nn.functional as F

# class InferenceBatchLogSoftmax(nn.Module):
#     def forward(self, input_):
#         batch_size = input_.size()[0]
#         return torch.stack([F.log_softmax(input_[i]) for i in range(batch_size)], 0)
    

class  CharCNN(nn.Module):
    
    def __init__(self, num_features):
        super(CharCNN, self).__init__()
        
        self.num_features = num_features
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(7, self.num_features), stride=1),
            nn.ReLU()
        )

        self.maxpool1 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))

        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(7, 256), stride=1),
            nn.ReLU()
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))

        self.conv3 = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(3, 256), stride=1),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(3, 256), stride=1),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(3, 256), stride=1),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(3, 256), stride=1),
            nn.ReLU()
        )

        self.maxpool6 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))

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
        self.fc3 =nn.Linear(1024, 4)
        self.softmax = nn.LogSoftmax()
            # nn.LogSoftmax()

        # self.inference_log_softmax = InferenceBatchLogSoftmax()

    def forward(self, x):
        debug=False
        x = x.unsqueeze(1)
        if debug:
            print('x.size()', x.size())

        x = self.conv1(x)
        if debug:
            print('x after conv1', x.size())

        x = x.transpose(1,3)
        if debug:
            print('x after transpose', x.size())

        x = self.maxpool1(x)
        if debug:
            print('x after maxpool1', x.size())

        x = self.conv2(x)
        if debug:
            print('x after conv2', x.size())

        x = x.transpose(1,3)
        if debug:
            print('x after transpose', x.size())

        x = self.maxpool2(x)
        if debug:
            print('x after maxpool2', x.size())

        x = self.conv3(x)
        if debug:
            print('x after conv3', x.size())

        x = x.transpose(1,3)
        if debug:
            print('x after transpose', x.size())


        x = self.conv4(x)
        if debug:
            print('x after conv4', x.size())

        x = x.transpose(1,3)
        if debug:
            print('x after transpose', x.size())


        x = self.conv5(x)
        if debug:
            print('x after conv5', x.size())

        x = x.transpose(1,3)
        if debug:
            print('x after transpose', x.size())


        x = self.conv6(x)
        if debug:
            print('x after conv6', x.size())

        x = x.transpose(1,3)
        if debug:
            print('x after transpose', x.size())


        x = self.maxpool6(x)
        if debug:
            print('x after maxpool6', x.size())

        x = x.view(x.size(0), -1)
        if debug:
            print('Collapse x:, ', x.size())

        x = self.fc1(x)
        if debug:
            print('FC1: ', x.size())

        x = self.fc2(x)
        if debug:
            print('FC2: ', x.size())

        x = self.fc3(x)
        if debug:
            print('x: ', x.size())

        x = self.softmax(x)
        # x = self.inference_log_softmax(x)

        return x