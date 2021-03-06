import torch
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels

def get_alexnet(pretrained):
    if pretrained:
        model = pretrainedmodels.__dict__["alexnet"](pretrained="imagenet")
    else:
        model = pretrainedmodels.__dict__["alexnet"](pretrained=None)

    print(model)

    model.last_linear = nn.Sequential(
        nn.BatchNorm1d(4096), nn.Dropout(p=0.25), 
        nn.Linear(in_features=4096, out_features=2048), nn.ReLU(),
        nn.BatchNorm1d(2048, eps=1e-5, momentum=0.1), nn.Dropout(p=0.5),
        nn.Linear(in_features=2048, out_features=1),
    )
    return model

def get_resnet(pretrained):
    if pretrained:
        model = pretrainedmodels.__dict__["resnet18"](pretrained="imagenet")
    else:
        model = pretrainedmodels.__dict__["resnet18"](pretrained=None)

    print(model)

    model.last_linear = nn.Sequential(
        nn.BatchNorm1d(512), nn.Dropout(p=0.25),
        nn.Linear(in_features=512, out_features=2048), nn.ReLU(),
        nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=2048, out_features=1),
    )
    return model


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # CNN part
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels= 256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        # Dense part
        self.fc1 = nn.Linear(in_features=9216, out_features=4096)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        # self.dropout2 = nn.Dropout(0.5)
        # self.fc3 = nn.Linear(in_features=4096, out_features=1000)
        # Last layer
        self.last_linear = nn.Sequential(
            nn.BatchNorm1d(4096), nn.Dropout(p=0.25), 
            nn.Linear(in_features=4096, out_features=2048), nn.ReLU(),
            nn.BatchNorm1d(2048, eps=1e-5, momentum=0.1), nn.Dropout(p=0.5),
            nn.Linear(in_features=2048, out_features=1),
        )


    def forward(self, image):
        # Image original size: (bs, 3, 227, 227)
        bs, c, h, w = image.size()
        x = F.relu(self.conv1(image)) # Size: (bs, 96, 55, 55)
        x = self.pool1(x) # (bs, 96, 27, 27)
        x = F.relu(self.conv2(x)) # (bs, 256, 27, 27)
        x = self.pool2(x) # (bs, 256, 13, 13)
        x = F.relu(self.conv3(x)) # (bs, 384, 13, 13)
        x = F.relu(self.conv4(x)) # (bs, 384, 13, 13)
        x = F.relu(self.conv5(x)) # (bs, 256, 13, 13)
        x = self.pool3(x) # (bs, 256, 6, 6)
        x = x.view(bs, -1) # (bs, 9216)
        x = F.relu(self.fc1(x)) # (bs, 4096)
        x = self.dropout1(x)
        x = F.relu(self.fc2(x)) # (bs, 4096)
        # x = self.dropout2(x)
        # x = F.relu(self.fc3(x)) # (bs, 1000)
        # x = torch.softmax(x, axis=1) # (bs, 1000)
        return self.last_linear(x)




