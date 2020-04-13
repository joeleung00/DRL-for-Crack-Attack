import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from functools import partial

class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size

conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)
conv1x1 = partial(Conv2dAuto, kernel_size=1, bias=False)

def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs),
                          'bn': nn.BatchNorm2d(out_channels) }))

def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv=conv3x3, activation="relu"):
        super().__init__()
        self.conv = conv
        self.activation = activation
        self.activate = activation_func(activation)
        self.in_channels, self.out_channels =  in_channels, out_channels
        self.blocks = nn.Sequential(
             conv_bn(self.in_channels, self.out_channels, self.conv),
             activation_func(self.activation),
             conv_bn(self.out_channels, self.out_channels, self.conv)
        )

    def forward(self, x):
        residual = x
        
        x = self.blocks(x)
        x += residual
        x = self.activate(x)

        return x

class ResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResidualBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'

        self.blocks = nn.Sequential(
            block(in_channels , out_channels, *args, **kwargs),
            *[block(out_channels, out_channels, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, blocks_size, activation="relu", block=ResidualBlock, *args, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.blocks_size = blocks_size
        self.activation = activation
        self.activate = activation_func(activation)
        self.gate = nn.Sequential(
            conv_bn(self.in_channels, self.blocks_size, conv3x3),
            activation_func(self.activation),
        )

        self.blocks = ResidualLayer(self.blocks_size, self.blocks_size,  n = 3)

        self.finalize = nn.Sequential(
             conv_bn(self.blocks_size, self.out_channels, conv1x1),
             activation_func(self.activation),
        )


    def forward(self, x):
        x = self.gate(x)
        x = self.blocks(x)
        x = self.finalize(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_features, n_classes, blocks_sizes):
        super().__init__()
        self.in_features = in_features
        self.n_classes = n_classes
        self.decode = nn.Sequential(
            nn.Linear(in_features, blocks_sizes[0]),
            *[nn.Linear(i, j) for (i, j) in zip(blocks_sizes, blocks_sizes[1:])],
            nn.Linear(blocks_sizes[-1], n_classes),
        )
        self.activate = activation_func("relu")



    def forward(self, x):
        x = self.decode(x.view(x.size()[0], -1))
        x = self.activate(x)
        return x

class ResidualNet(nn.Module):
    def __init__(self, in_channels, n_classes, flatten_board_size, encoder_blocks_size, 
                 decoder_blocks_sizes=[256], *args, **kwargs):
        super().__init__()
        self.encoder = Encoder(in_channels, 1, encoder_blocks_size, *args, **kwargs)
        self.decoder = Decoder(flatten_board_size, n_classes, decoder_blocks_sizes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == "__main__":
    net = ResidualNet(4, 10, 128, 36)
    print(net)
