import torch.nn as nn
import torch
import torch.nn.functional as F

class Conv_layer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, cardinality=1):
        super(Conv_layer,self).__init__()
        self.layer = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, groups=cardinality),
                                   nn.BatchNorm2d(out_channel),
                                   nn.ReLU())
    
    def forward(self, x):
        return self.layer(x)

class ResNeXt_block(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, cardinality=32):
        super(ResNeXt_block,self).__init__()
        block = [Conv_layer(in_channel, out_channel//2, kernel_size=1, padding=0),
                 Conv_layer(out_channel//2, out_channel//2, stride=stride, cardinality=cardinality),
                 Conv_layer(out_channel//2, out_channel, kernel_size=1, padding=0)]
        self.ResNeXt_block = nn.Sequential(*block)
        if in_channel!=out_channel or stride!=1:
            self.shorcut = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride),
                                         nn.BatchNorm2d(out_channel),
                                         nn.ReLU())
        else:
            self.shorcut = nn.Sequential()
    
    def forward(self, x):
        return self.ResNeXt_block(x) + self.shorcut(x)

class ResNeXt_stage(nn.Module):
    def __init__(self, in_channel, out_channel, num_layer, stride=1, cardinality=32):
        super(ResNeXt_stage,self).__init__()
        stage = []
        stage.append(ResNeXt_block(in_channel, out_channel, stride=stride, cardinality=cardinality))
        for _ in range(num_layer-1):
            stage.append(ResNeXt_block(out_channel, out_channel, cardinality=cardinality))
        self.ResNeXt_stage = nn.Sequential(*stage)

    def forward(self, x):
        return self.ResNeXt_stage(x)

class ResNeXt(nn.Module):
    def __init__(self, num_classes=10, num_layers=[3,4,6,3], cardinality=32):
        super(ResNeXt,self).__init__()
        self.num_classes = num_classes
        self.stage1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                                    nn.MaxPool2d(kernel_size=3, stride=2))
        self.stage2 = ResNeXt_stage(64, 256, num_layers[0], cardinality=cardinality)
        self.stage3 = ResNeXt_stage(256, 512, num_layers[1], stride=2, cardinality=cardinality)
        self.stage4 = ResNeXt_stage(512, 1024, num_layers[2], stride=2, cardinality=cardinality)
        self.stage5 = ResNeXt_stage(1024, 2048, num_layers[3], stride=2, cardinality=cardinality)
        self.fc_layer = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(2048, self.num_classes, kernel_size=1))

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.fc_layer(x).squeeze()
        return x