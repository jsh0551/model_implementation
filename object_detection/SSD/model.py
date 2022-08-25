import torchvision
import torch.nn as nn
import torch
import torch.nn.functional as F

class CNN_layer(nn.Module):
    def __init__(self, in_channel, out_channel, num_layers):
        super().__init__()
        start_block = [nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
                           nn.ReLU()]
        blocks = []
        for _ in range(num_layers-1):
            blocks.append(nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)),
            blocks.append(nn.ReLU())
        self.blocks = nn.Sequential(*(start_block + blocks))
    def forward(self,x):
        return self.blocks(x)

class SSD_CNN_layer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.extra_layer = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
                                         nn.BatchNorm2d(out_channel),
                                         nn.ReLU())
    def forward(self,x):
        return self.extra_layer(x)


class VGG_Architecture(nn.Module):
    def __init__(self, nums_layers = [2,2,3,3,3]):
        super(VGG_Architecture,self).__init__()
        self.nums_layers = nums_layers
        self.layer1 = CNN_layer(3, 64, self.nums_layers[0])
        self.Max1 = nn.MaxPool2d(2,2, ceil_mode=True)
        self.layer2 = CNN_layer(64, 128, self.nums_layers[1])
        self.Max2 = nn.MaxPool2d(2,2, ceil_mode=True)
        self.layer3 = CNN_layer(128,256,self.nums_layers[2])
        self.Max3 = nn.MaxPool2d(2,2, ceil_mode=True)
        self.layer4 = CNN_layer(256,512,self.nums_layers[3])
        self.Max4 = nn.MaxPool2d(2,2, ceil_mode=True)
        self.layer5 = CNN_layer(512,512,self.nums_layers[4])

        self.load_pretrained()
        

    def forward(self,x):
        x = self.layer1(x)
        x = self.Max1(x)
        x = self.layer2(x)
        x = self.Max2(x)
        x = self.layer3(x)
        x = self.Max3(x)
        x = self.layer4(x)
        feature_map = x

        x = self.Max4(x)
        output = self.layer5(x)
        return feature_map, output

    def load_pretrained(self):
        state_dict = self.state_dict()
        pretrained_dict = torchvision.models.vgg16(pretrained=True).state_dict()
        state_list = list(state_dict.keys())
        pretrained_list = list(pretrained_dict.keys())

        for i,pretrained_weight in enumerate(pretrained_list[:-6]):
            state_dict[state_list[i]] = pretrained_dict[pretrained_weight]

        self.load_state_dict(state_dict)
        print('load pretrained model.')


class SSD_CNN_layers(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer6 = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=6, dilation=6),
                      nn.BatchNorm2d(1024),
                      nn.ReLU())
        self.layer7 = SSD_CNN_layer(1024,1024, kernel_size=1)

        self.layer8_1 = SSD_CNN_layer(1024,256, kernel_size=1)
        self.layer8_2 = SSD_CNN_layer(256,512, kernel_size=3, stride=2, padding=1)

        self.layer9_1 = SSD_CNN_layer(512,128, kernel_size=1)
        self.layer9_2 = SSD_CNN_layer(128,256, kernel_size=3, stride=2, padding=1)

        self.layer10_1 = SSD_CNN_layer(256,128, kernel_size=1)
        self.layer10_2 = SSD_CNN_layer(128,256, kernel_size=3, padding=0)

        self.layer11_1 = SSD_CNN_layer(256,128, kernel_size=1)
        self.layer11_2 = SSD_CNN_layer(128,256, kernel_size=3, padding=0)

    def forward(self,x):
        x = self.layer6(x)
        x = self.layer7(x)
        feature_map2 = x

        x = self.layer8_1(x)
        x = self.layer8_2(x)
        feature_map3 = x
        
        x = self.layer9_1(x)
        x = self.layer9_2(x)
        feature_map4 = x

        x = self.layer10_1(x)
        x = self.layer10_2(x)
        feature_map5 = x

        x = self.layer11_1(x)
        x = self.layer11_2(x)
        feature_map6 = x

        return feature_map2, feature_map3, feature_map4, feature_map5, feature_map6


class Convolution_to_lc_cf(nn.Module):
    def __init__(self, num_classes = 10, num_bboxes = [4,6,6,6,4,4]):
        super().__init__()
        self.num_bboxes = num_bboxes
        self.num_classes = num_classes+1
        self.ft_conv1 = nn.Conv2d(512,self.num_bboxes[0]*(4 + self.num_classes), kernel_size=3, padding=1)
        self.ft_conv2 = nn.Conv2d(1024,self.num_bboxes[1]*(4 + self.num_classes), kernel_size=3, padding=1)
        self.ft_conv3 = nn.Conv2d(512,self.num_bboxes[2]*(4 + self.num_classes), kernel_size=3, padding=1)
        self.ft_conv4 = nn.Conv2d(256,self.num_bboxes[3]*(4 + self.num_classes), kernel_size=3, padding=1)
        self.ft_conv5 = nn.Conv2d(256,self.num_bboxes[4]*(4 + self.num_classes), kernel_size=3, padding=1)
        self.ft_conv6 = nn.Conv2d(256,self.num_bboxes[5]*(4 + self.num_classes), kernel_size=3, padding=1)

    def get_lc_cf(self, ft_conv, ft_map, num_bbox):
        conv_ft_map = ft_conv(ft_map)
        batch_size = ft_map.shape[0]

        ft_map_lc = conv_ft_map[..., :num_bbox*4, :, :].permute(0,2,3,1).contiguous()
        lc = ft_map_lc.view(batch_size,-1,4)

        ft_map_cf = conv_ft_map[..., num_bbox*4:, :, :].permute(0,2,3,1).contiguous()
        cf = ft_map_cf.view(batch_size,-1,self.num_classes)

        return lc, cf

    def forward(self, ft_map1, ft_map2, ft_map3, ft_map4, ft_map5, ft_map6):
        lc1, cf1 = self.get_lc_cf(self.ft_conv1, ft_map1, self.num_bboxes[0])
        lc2, cf2 = self.get_lc_cf(self.ft_conv2, ft_map2, self.num_bboxes[1])
        lc3, cf3 = self.get_lc_cf(self.ft_conv3, ft_map3, self.num_bboxes[2])
        lc4, cf4 = self.get_lc_cf(self.ft_conv4, ft_map4, self.num_bboxes[3])
        lc5, cf5 = self.get_lc_cf(self.ft_conv5, ft_map5, self.num_bboxes[4])
        lc6, cf6 = self.get_lc_cf(self.ft_conv6, ft_map6, self.num_bboxes[5])

        localization = torch.cat([lc1,lc2,lc3,lc4,lc5,lc6], dim=-2)
        confidence = torch.cat([cf1,cf2,cf3,cf4,cf5,cf6], dim=-2)
        return localization, confidence


class SSD(nn.Module):
    def __init__(self, num_classes = 10):
        super(SSD,self).__init__()
        self.vgg_layer = VGG_Architecture()
        self.ssd_layer = SSD_CNN_layers()
        self.conv_to_lc_cf = Convolution_to_lc_cf(num_classes=num_classes, num_bboxes = [4,6,6,6,4,4])

    def forward(self,x):
        ft_map1, vgg_output = self.vgg_layer(x)
        ft_map2, ft_map3, ft_map4, ft_map5, ft_map6 = self.ssd_layer(vgg_output)
        lc,cf = self.conv_to_lc_cf(ft_map1, ft_map2, ft_map3, ft_map4, ft_map5, ft_map6)

        return lc,cf
        