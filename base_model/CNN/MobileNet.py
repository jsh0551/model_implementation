from turtle import forward
import torch.nn as nn
import torch
import torch.nn.functional as F

class MobileNet_layer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super(MobileNet_layer, self).__init__()
        self.depthwise_layer = nn.Sequential(nn.BatchNorm2d(in_channel),
                                             nn.ReLU(),
                                             nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channel))
        self.pointwise_layer = nn.Sequential(nn.BatchNorm2d(in_channel),
                                             nn.ReLU(),
                                             nn.Conv2d(in_channel, out_channel, kernel_size=1))
    def forward(self, x):
        x = self.depthwise_layer(x)
        x = self.pointwise_layer(x)
        return x

class MobileNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNet, self).__init__()
        self.num_classes = num_classes
        self.layer1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.layer2 = nn.Sequential(*[MobileNet_layer(32, 64)],*[MobileNet_layer(64, 128, stride=2)])
        self.layer3 = nn.Sequential(*[MobileNet_layer(128, 128)],*[MobileNet_layer(128, 256, stride=2)])
        self.layer4 = nn.Sequential(*[MobileNet_layer(256, 256)],*[MobileNet_layer(256, 512, stride=2)])
        layer5 = [MobileNet_layer(512,512) for _ in range(5)]
        self.layer5 = nn.Sequential(*layer5)
        self.layer6 = nn.Sequential(*[MobileNet_layer(512, 1024, stride=2)],*[MobileNet_layer(1024, 1024, stride=2)])
        self.fc_layer = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                      nn.Conv2d(1024, num_classes, kernel_size=1))
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.fc_layer(x).squeeze()

        return x