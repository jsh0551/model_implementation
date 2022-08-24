import torch.nn as nn
import torch
import torch.nn.functional as F

class CNN_layer(nn.Module):
    def __init__(self, in_channel, out_channel, num_layers):
        super().__init__()
        start_block = [nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
                       nn.BatchNorm2d(out_channel),
                           nn.ReLU()]
        blocks = []
        for _ in range(num_layers-1):
            blocks.append(nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1))
            blocks.append(nn.BatchNorm2d(out_channel))
            blocks.append(nn.ReLU())
        self.blocks = nn.Sequential(*(start_block + blocks))
    def forward(self,x):
        return self.blocks(x)


class VGG_Architecture(nn.Module):
    def __init__(self, num_classes = 10, nums_layers = [2,2,3,3,3]):
        super(VGG_Architecture,self).__init__()
        self.nums_layers = nums_layers
        self.num_classes = num_classes
        self.layer1 = CNN_layer(3, 64, self.nums_layers[0])
        self.Max1 = nn.MaxPool2d(2,2, ceil_mode=True)
        self.layer2 = CNN_layer(64, 128, self.nums_layers[1])
        self.Max2 = nn.MaxPool2d(2,2, ceil_mode=True)
        self.layer3 = CNN_layer(128,256,self.nums_layers[2])
        self.Max3 = nn.MaxPool2d(2,2, ceil_mode=True)
        self.layer4 = CNN_layer(256,512,self.nums_layers[3])
        self.Max4 = nn.MaxPool2d(2,2, ceil_mode=True)
        self.layer5 = CNN_layer(512,512,self.nums_layers[4])
        self.Max5 = nn.MaxPool2d(2,2, ceil_mode=True)

        self.fc = nn.Sequential(nn.Flatten(),
                                nn.Linear(7*7*512,4096),
                                nn.ReLU(),
                                # nn.Dropout(p=0.2),
                                nn.Linear(4096,4096),
                                nn.ReLU(),
                                nn.Linear(4096,self.num_classes))
    def forward(self,x):
        
        x = self.layer1(x)
        x = self.Max1(x)
        x = self.layer2(x)
        x = self.Max2(x)
        x = self.layer3(x)
        x = self.Max3(x)
        x = self.layer4(x)
        x = self.Max4(x)
        x = self.layer5(x)
        x = self.Max5(x)
        x = self.fc(x)
        # output = F.softmax(x)

        return x

class VGGNet(nn.Module):
    def __init__(self, num_classes=10, num_layers=16):
        super(VGGNet,self).__init__()
        if num_layers == 11:
            self.model =  VGG_Architecture(num_classes=num_classes, nums_layers=[1,1,2,2,2])
        elif num_layers == 13:
            self.model =  VGG_Architecture(num_classes=num_classes, nums_layers=[2,2,2,2,2])
        elif num_layers == 19:
            self.model =  VGG_Architecture(num_classes=num_classes, nums_layers=[2,2,4,4,4])
        else:
            self.model = VGG_Architecture(num_classes=num_classes)
    def forward(self,x):
        return self.model(x)