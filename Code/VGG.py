# Reference: Dive into deep learning
import torch.nn as nn

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs):
        super(VGGBlock, self).__init__()
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.vgg_block = nn.Sequential(*layers)   # Sequential是把各个层连接在一起

    def forward(self, x):
        return self.vgg_block(x)

class VGG(nn.Module):
    def _init_(self,conv_arch,in_channels):  # conv_arch是一个列表，里面装着每一个VGG块的参数，形如（10,12）
        super(VGG, self).__init__()
        conv_blks = []
        for (num_convs, out_channels) in conv_arch:
            block = VGGBlock(in_channels,out_channels,num_convs)
            conv_blks.append(block)
            in_channels = out_channels

        self.vgg_network = nn.Sequential(
            *conv_blks, nn.Flatten(),
            nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 1000) )
        
    def forward(self,x):
        return self.vgg_network(x)
