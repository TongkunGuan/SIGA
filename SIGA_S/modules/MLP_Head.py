import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.Fusion_Package import GatedBimodal
import numpy as np


class MlpBlock(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(MlpBlock, self).__init__()
        self.mlp_dim = input_size
        self.hid_dim = hidden_size
        self.Dense0 = nn.Linear(self.mlp_dim, self.hid_dim)
        self.activate = nn.GELU()
        self.Dense1 = nn.Linear(self.hid_dim, self.mlp_dim)

    def forward(self, x):
        y = self.Dense0(x)
        y = self.activate(y)
        y = self.Dense1(y)
        return y


class MixerBlock(nn.Module):

    def __init__(self, Dim_size, Patch_size):
        super(MixerBlock, self).__init__()
        self.Dim_size = Dim_size
        self.Patch_size = Patch_size
        self.MlpBlock_Patches = MlpBlock(Patch_size, Patch_size * 2)
        self.MlpBlock_Channels = MlpBlock(Dim_size, Dim_size * 2)
        self.LN = nn.LayerNorm(Dim_size)

    def forward(self, x):
        nB, nS, nC = x.shape
        y = self.LN(x)
        y = y.permute(0, 2, 1)
        y = self.MlpBlock_Patches(y)
        y = y.permute(0, 2, 1)
        x = x + y
        y = self.LN(x)
        return x + self.MlpBlock_Channels(y)


class MLP(nn.Module):

    def __init__(self, input_size, hidden_size, num_blocks):
        super(MLP, self).__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1))
        self.conv = nn.Conv2d(input_size, hidden_size, kernel_size=(8, 2), stride=(8, 2), padding=(0, 0), bias=False)

        self.num_blocks = num_blocks
        self.output_size = 25
        layers = []
        for _ in range(self.num_blocks):
            layers.append(MixerBlock(hidden_size, self.output_size))
        self.MixerBlocks = nn.Sequential(*layers)
        self.LN = nn.LayerNorm([self.output_size, hidden_size])

    def forward(self, inputs):
        # x = self.conv1(inputs)
        x = self.maxpool1(inputs)
        # x = self.conv2(x)
        x = self.conv(x)

        nB, nC, nH, nW = x.shape
        x = x.view(nB, nC, -1).permute(0, 2, 1)
        for i in range(self.num_blocks):
            x = self.MixerBlocks[i](x)
        x = self.LN(x)  # [nB, 100, nC]
        x = x.mean(axis=1)
        return x

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 3
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale

class Multi_MLP(nn.Module):

    def __init__(self, input_size, hidden_size, input_H=16, input_W=50, num_blocks=1, iterable=1):
        super(Multi_MLP, self).__init__()
        if iterable == 1:
            Kernel_size = [(4, 1), (4, 2), (4, 4)]
        elif iterable == 2:
            Kernel_size = [(8, 2), (8, 4), (8, 8)]
        elif iterable == 3:
            Kernel_size = [(2, 1), (2, 2), (2, 3)]

        # self.conv1 = nn.Sequential(nn.Conv2d(input_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=False),
        #                            nn.BatchNorm2d(hidden_size),
        #                            nn.ReLU(inplace=True))
        self.SpatialGate = SpatialGate()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1))
        self.conv2 = nn.Sequential(nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(hidden_size),
                                   nn.ReLU(inplace=True))

        self.num_blocks = num_blocks

        Layers = []
        Conv_layers = []
        for Kernel in Kernel_size:

            Conv_layers.append(nn.Conv2d(input_size, hidden_size, kernel_size=Kernel, stride=Kernel,
                                         padding=(0, 0), bias=False))

            MixerBlock_layers = []
            self.Patch_size = np.floor((input_W + 1) / Kernel[1]).astype(np.int)
            self.Channel_size = hidden_size
            for _ in range(self.num_blocks):
                MixerBlock_layers.append(MixerBlock(self.Channel_size, self.Patch_size))

            MixerBlock_layers.append(nn.LayerNorm([self.Patch_size, self.Channel_size]))

            Layers.append(nn.Sequential(*MixerBlock_layers))

        self.Conv = nn.Sequential(*Conv_layers)
        self.MixerBlocks = nn.Sequential(*Layers)

        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.SelectAttention = nn.Linear(3 * hidden_size, hidden_size)
        self.LN = nn.LayerNorm(hidden_size)

    def forward(self, inputs):
        # x = self.conv1(inputs)
        # x = self.SpatialGate(inputs)
        x = self.maxpool1(inputs)  # (H/2, W+1)
        # x = self.conv2(x)
        result = []
        for i in range(3):
            y = self.Conv[i](x)
            nB, nC, nH, nW = y.shape
            y = y.view(nB, nC, -1).permute(0, 2, 1)
            y = self.MixerBlocks[i](y)
            y = self.LN(y)
            e = self.score(torch.tanh(y))  # batch_size x num_encoder_step * 1
            alpha = F.softmax(e, dim=1)
            Attented_Seq = torch.bmm(alpha.permute(0, 2, 1), y).squeeze(1)  # batch_size x num_channel
            result.append(Attented_Seq)
        out = self.SelectAttention(torch.cat(result, dim=-1))
        return out


class MultiModal_Fusion(nn.Module):

    def __init__(self, input_dim, hidden_dim, input_H, input_W, num_blocks, iterable=1):
        super(MultiModal_Fusion, self).__init__()
        # self.GatedBimodal = GatedBimodal(input_dim)
        self.Gate_Head = nn.Linear(input_dim * 2, hidden_dim)
        self.MLP_Head = Multi_MLP(input_dim, hidden_dim, input_H, input_W, num_blocks, iterable)
        self.Cat_Head = nn.Linear(input_dim * 2, hidden_dim)

    def Global_Fusion(self, Contextual_Feature2D, Feature1D):
        Global_0 = Contextual_Feature2D
        Global_1 = Feature1D
        # output = self.GatedBimodal(Global_1, Global_0)
        output = self.Gate_Head(torch.cat([Global_1, Global_0], dim=-1))
        return output

    def Dense_Fusion(self, Feature2D, Feature1D):
        nB, nC, nH, nW = Feature2D.shape
        Feature1D = Feature1D.unsqueeze(2).unsqueeze(2)
        Dense_0 = Feature1D.repeat(1, 1, nH, nW)
        Dense_1 = Feature2D
        Dense_Feature = Dense_0 + Dense_1
        output = self.MLP_Head(Dense_Feature)
        return output

    def forward(self, Contextual_Feature2D, Feature2D, Feature1D):
        if len(Feature2D.shape) == 5:
            nB, nT, nC, nH, nW = Feature2D.shape
            Global_ = self.Global_Fusion(Contextual_Feature2D, Feature1D)
            Dense_ = self.Dense_Fusion(Feature2D.view(nB * nT, nC, nH, nW), Feature1D.view(nB * nT, -1))
            Dense = Dense_.view(nB, nT, nC)
        else:
            nB, nC, nH, nW = Feature2D.shape
            Global_ = self.Global_Fusion(Contextual_Feature2D, Feature1D)
            Dense_ = self.Dense_Fusion(Feature2D.view(nB, nC, nH, nW), Feature1D.view(nB, -1))
            Dense = Dense_.view(nB, nC)
        Encode_f = torch.cat([Global_, Dense], dim=-1)
        output = self.Cat_Head(Encode_f)
        return output

#
# M = MLP(128, 128, 2)
# print(M)
# input = torch.randn(2,128,16,50)
# M(input)

# input = torch.randn(2, 256, 4, 12)
# M = Multi_MLP(256, 256, input.shape[2], input.shape[3], 1, 3)
# A = M(input)
# print(A.shape)
