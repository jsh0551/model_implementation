import torch.nn as nn
import torch
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_channel, growth_rate = 32, Bottle_Neck = True):
        super(Block, self).__init__()
        if Bottle_Neck:
            self.block = nn.Sequential(nn.BatchNorm2d(in_channel), nn.ReLU(),
                                       nn.Conv2d(in_channel, growth_rate*4, kernel_size=1),
                                       nn.BatchNorm2d(growth_rate*4), nn.ReLU(),
                                       nn.Conv2d(growth_rate*4, growth_rate, kernel_size=3, padding=1))
        else:
            
            self.block = nn.Sequential(nn.BatchNorm2d(in_channel), nn.ReLU(),
                                       nn.Conv2d(in_channel, growth_rate, kernel_size=3, padding=1))
    
    def forward(self,x):
        return self.block(x)


class Dense_block(nn.Module):
    def __init__(self, in_channel, num_layer, growth_rate = 32, Bottle_Neck = True):
        super(Dense_block,self).__init__()
        dense_block = []
        self.num_layer = num_layer
        for l in range(self.num_layer):
            tmp_block = Block(in_channel=in_channel + growth_rate*l, growth_rate=growth_rate, Bottle_Neck=Bottle_Neck)
            dense_block.append(tmp_block)
        self.dense_block = nn.Sequential(*dense_block)

    def forward(self,x):
        dense_list = []
        for l in range(self.num_layer):
            dense_list.append(x)
            iuput_x = torch.cat(dense_list,dim=1)
            x = self.dense_block[l](iuput_x)

        return torch.cat(dense_list, dim=1)

class Transition_block(nn.Module):
    def __init__(self, in_channel, compression_rate = 0.5):
        super(Transition_block,self).__init__()
        self.transition_block = nn.Sequential(nn.BatchNorm2d(in_channel),
                                nn.Conv2d(in_channel,int(in_channel*compression_rate),kernel_size=1),
                                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
    def forward(self,x):
        return self.transition_block(x)



class DenseNet(nn.Module):
    def __init__(self, num_classes = 10, num_layers = [6,12,24,16], growth_rate = 32, Bottle_Neck = True, compression_rate = 0.5):
        super(DenseNet,self).__init__()
        self.first_layer = nn.Sequential(nn.Conv2d(3, growth_rate*2, kernel_size=7, stride=2, padding=3),
                                         nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True))
        self.num_classes = num_classes
        dense_block_list =[]
        tmp_in_channel = growth_rate*2

        for i,n in enumerate(num_layers):
            tmp_dense_block = Dense_block(tmp_in_channel, n, growth_rate = growth_rate, Bottle_Neck = Bottle_Neck)
            dense_block_list.append(tmp_dense_block)
            tmp_in_channel = (n-1)*growth_rate + tmp_in_channel

            if i < (len(num_layers)-1):
                tmp_transition_block = Transition_block(tmp_in_channel, compression_rate=compression_rate)
                tmp_in_channel = int(tmp_in_channel*compression_rate)
                dense_block_list.append(tmp_transition_block)

            else:
                self.fc_layer = nn.Sequential(nn.BatchNorm2d(tmp_in_channel),
                                nn.AdaptiveAvgPool2d(7),
                                nn.Conv2d(tmp_in_channel, num_classes, kernel_size=7))

        self.densenet_blocks = nn.Sequential(*dense_block_list)

    def forward(self,x):
        x = self.first_layer(x)
        x = self.densenet_blocks(x)
        output = self.fc_layer(x)
        return output.view(-1,self.num_classes)