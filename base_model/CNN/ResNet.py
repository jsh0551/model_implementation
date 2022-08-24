import torch.nn as nn
import torch
import torch.nn.functional as F

def padding_identity(x):
    reduced_x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
    padding_zeros = torch.zeros_like(reduced_x)
    return torch.cat([reduced_x,padding_zeros],dim=1)

class block(nn.Module):
    def __init__(self, in_channel, out_channel, start_stride = 1, bottle_neck = False, reduction = False):
        super().__init__()
        self.reduction = reduction
        self.scale = out_channel//in_channel
        if not bottle_neck:
            self.block = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=start_stride, padding=1),
                        nn.BatchNorm2d(out_channel),
                        nn.ReLU(),
                        nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
                        nn.BatchNorm2d(out_channel),
                        nn.ReLU())
                        
        else: # bottle neck block
            self.block = nn.Sequential(nn.Conv2d(self.scale*out_channel, out_channel, kernel_size=1, stride=start_stride),
                        nn.BatchNorm2d(out_channel),
                        nn.ReLU(),
                        nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
                        nn.BatchNorm2d(out_channel),
                        nn.ReLU(),
                        nn.Conv2d(out_channel, 4*out_channel, kernel_size=1),
                        nn.BatchNorm2d(4*out_channel),
                        nn.ReLU()
                        )
                        

    def forward(self, x):
        if self.reduction:
            identity = padding_identity(x)
            x = self.block(x)
            x = x + F.relu(identity)
        else:
            identity = x
            x = self.block(x)
            x = x + F.relu(identity)      

        return x
    
        
class ResNet(nn.Module):
    def __init__(self, num_layers = 34, num_blocks = [3,4,6,3], num_classes = 10, bottle_neck = False):
        super(ResNet, self).__init__()
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.conv_layer1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                                         nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        block2 = block(64, 64, bottle_neck = bottle_neck)
        block3 = block(128, 128, bottle_neck = bottle_neck)
        block4 = block(256, 256, bottle_neck = bottle_neck)
        block5 = block(512, 512, bottle_neck = bottle_neck)

        self.conv_layer2 = nn.Sequential(*[block2 for _ in range(num_blocks[0])])

        self.conv_layer3_start = block(64, 128, start_stride=2, bottle_neck = bottle_neck, reduction = True)
        self.conv_layer3 = nn.Sequential(*[block3 for _ in range(num_blocks[1]-1)])

        self.conv_layer4_start = block(128, 256, start_stride=2, bottle_neck = bottle_neck, reduction = True)
        self.conv_layer4 = nn.Sequential(*[block4 for _ in range(num_blocks[2]-1)])

        self.conv_layer5_start = block(256, 512, start_stride=2, bottle_neck = bottle_neck, reduction = True)
        self.conv_layer5 = nn.Sequential(*[block5 for _ in range(num_blocks[3]-1)])
        self.dropout = nn.Dropout(p=0.2)
        try:
            self.avgpool = nn.AdaptiveAvgPool2d(7)
            if bottle_neck:
                self.fcn = nn.Conv2d(2048, num_classes, kernel_size=7)
            else:
                self.fcn = nn.Conv2d(512, num_classes, kernel_size=7)
        except:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            if bottle_neck:
                self.fcn = nn.Conv2d(2048, num_classes, kernel_size=1)
            else:
                self.fcn = nn.Conv2d(512, num_classes, kernel_size=1)
        

    def forward(self,x):
        x = self.conv_layer1(x)

        x = self.conv_layer2(x)

        x = self.conv_layer3_start(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4_start(x)
        x = self.conv_layer4(x)
        x = self.conv_layer5_start(x)
        x = self.conv_layer5(x)
        x = self.dropout(x)
        x = self.avgpool(x)
        x = self.fcn(x)
        x = x.view(-1,self.num_classes)
        # pred = F.softmax(x,dim=1).view(-1,self.num_classes)
        return x

    def how_to_use(self):
        print("ResNet(num_layers, num_blocks, num_classes, bottle_neck)\n")
        print(" - num_layers : 18, 34, 50, 101, 152")
        print(" - num_blocks : [2,2,2,2], [3,4,6,3], [3,4,6,3], [3,4,23,3], [3,8,36,3]")
        print("   (num_layers and num_blocks should match orderly each other)")
        print(" - num_classes : number of classification categories")
        print(" - bottle_neck : True -> use bottle neck, False -> not use bottle neck")