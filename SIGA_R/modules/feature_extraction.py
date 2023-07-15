import torch.nn as nn
import torch.nn.functional as F
import math

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class VGG_FeatureExtractor(nn.Module):
    """FeatureExtractor of CRNN (https://arxiv.org/pdf/1507.05717.pdf)"""

    def __init__(self, input_channel, output_channel=512):
        super(VGG_FeatureExtractor, self).__init__()
        self.output_channel = [
            int(output_channel / 8),
            int(output_channel / 4),
            int(output_channel / 2),
            output_channel,
        ]  # [64, 128, 256, 512]
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 64x16x50
            nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 128x8x25
            nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1),
            nn.ReLU(True),  # 256x8x25
            nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 256x4x25
            nn.Conv2d(
                self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False
            ),
            nn.BatchNorm2d(self.output_channel[3]),
            nn.ReLU(True),  # 512x4x25
            nn.Conv2d(
                self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False
            ),
            nn.BatchNorm2d(self.output_channel[3]),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 512x2x25
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0),
            nn.ReLU(True),
        )  # 512x1x24

    def forward(self, input):
        return self.ConvNet(input)


class RCNN_FeatureExtractor(nn.Module):
    """FeatureExtractor of GRCNN (https://papers.nips.cc/paper/6637-gated-recurrent-convolution-neural-network-for-ocr.pdf)"""

    def __init__(self, input_channel, output_channel=512):
        super(RCNN_FeatureExtractor, self).__init__()
        self.output_channel = [
            int(output_channel / 8),
            int(output_channel / 4),
            int(output_channel / 2),
            output_channel,
        ]  # [64, 128, 256, 512]
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 64 x 16 x 50
            GRCL(
                self.output_channel[0],
                self.output_channel[0],
                num_iteration=5,
                kernel_size=3,
                pad=1,
            ),
            nn.MaxPool2d(2, 2),  # 64 x 8 x 25
            GRCL(
                self.output_channel[0],
                self.output_channel[1],
                num_iteration=5,
                kernel_size=3,
                pad=1,
            ),
            nn.MaxPool2d(2, (2, 1), (0, 1)),  # 128 x 4 x 26
            GRCL(
                self.output_channel[1],
                self.output_channel[2],
                num_iteration=5,
                kernel_size=3,
                pad=1,
            ),
            nn.MaxPool2d(2, (2, 1), (0, 1)),  # 256 x 2 x 27
            nn.Conv2d(
                self.output_channel[2], self.output_channel[3], 2, 1, 0, bias=False
            ),
            nn.BatchNorm2d(self.output_channel[3]),
            nn.ReLU(True),
        )  # 512 x 1 x 26

    def forward(self, input):
        return self.ConvNet(input)

# For Gated RCNN
class GRCL(nn.Module):
    def __init__(self, input_channel, output_channel, num_iteration, kernel_size, pad):
        super(GRCL, self).__init__()
        self.wgf_u = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False)
        self.wgr_x = nn.Conv2d(output_channel, output_channel, 1, 1, 0, bias=False)
        self.wf_u = nn.Conv2d(
            input_channel, output_channel, kernel_size, 1, pad, bias=False
        )
        self.wr_x = nn.Conv2d(
            output_channel, output_channel, kernel_size, 1, pad, bias=False
        )

        self.BN_x_init = nn.BatchNorm2d(output_channel)

        self.num_iteration = num_iteration
        self.GRCL = [GRCL_unit(output_channel) for _ in range(num_iteration)]
        self.GRCL = nn.Sequential(*self.GRCL)

    def forward(self, input):
        """The input of GRCL is consistant over time t, which is denoted by u(0)
        thus wgf_u / wf_u is also consistant over time t.
        """
        wgf_u = self.wgf_u(input)
        wf_u = self.wf_u(input)
        x = F.relu(self.BN_x_init(wf_u))

        for i in range(self.num_iteration):
            x = self.GRCL[i](wgf_u, self.wgr_x(x), wf_u, self.wr_x(x))

        return x


class GRCL_unit(nn.Module):
    def __init__(self, output_channel):
        super(GRCL_unit, self).__init__()
        self.BN_gfu = nn.BatchNorm2d(output_channel)
        self.BN_grx = nn.BatchNorm2d(output_channel)
        self.BN_fu = nn.BatchNorm2d(output_channel)
        self.BN_rx = nn.BatchNorm2d(output_channel)
        self.BN_Gx = nn.BatchNorm2d(output_channel)

    def forward(self, wgf_u, wgr_x, wf_u, wr_x):
        G_first_term = self.BN_gfu(wgf_u)
        G_second_term = self.BN_grx(wgr_x)
        G = F.sigmoid(G_first_term + G_second_term)

        x_first_term = self.BN_fu(wf_u)
        x_second_term = self.BN_Gx(self.BN_rx(wr_x) * G)
        x = F.relu(x_first_term + x_second_term)

        return x
class ResNet_FeatureExtractor(nn.Module):
    """FeatureExtractor of FAN (http://openaccess.thecvf.com/content_ICCV_2017/papers/Cheng_Focusing_Attention_Towards_ICCV_2017_paper.pdf)"""

    def __init__(self, input_channel, output_channel=512):
        super(ResNet_FeatureExtractor, self).__init__()
        # self.ConvNet = ResNet_31(input_channel, output_channel, BasicBlock, [1, 2, 5, 3])
        self.ConvNet = ResNet_45(BasicBlock, [3, 4, 6, 6, 3])

    def forward(self, input):
        return self.ConvNet(input)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet_45(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 32
        super(ResNet_45, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 32, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=1)
        self.layer5 = self._make_layer(block, 512, layers[4], stride=1)

        self.maxpool6 = nn.MaxPool2d(kernel_size=(2, 3), stride=(2, 1), padding=(0, 1))
        self.layer6 = self._make_layer(
            block, 512, 2, stride=1
        )
        self.conv6 = nn.Conv2d(
            512,
            512,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn6 = nn.BatchNorm2d(512)

        self.layer7 = self._make_layer(
            block, 512, 1, stride=1
        )
        self.conv7_1 = nn.Conv2d(
            512,
            512,
            kernel_size=(2, 3),
            stride=(2, 1),
            padding=(0, 1),
            bias=False,
        )
        self.bn7_1 = nn.BatchNorm2d(512)
        self.conv7_2 = nn.Conv2d(
            512,
            512,
            kernel_size=(2, 3),
            stride=1,
            padding=(0, 1),
            bias=False,
        )
        self.bn7_2 = nn.BatchNorm2d(512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        FPN = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # torch.Size([256, 32, 32, 128])
        FPN.append(x)
        x = self.layer1(x)  # torch.Size([256, 32, 16, 64])
        x = self.layer2(x)  # torch.Size([256, 64, 16, 64])
        FPN.append(x)
        x = self.layer3(x)  # torch.Size([256, 128, 8, 32])
        x = self.layer4(x)  # torch.Size([256, 256, 8, 32])
        FPN.append(x)
        x = self.layer5(x)  # torch.Size([256, 512, 8, 32])

        """ Build 1D Features"""
        x = self.maxpool6(x) #torch.Size([256, 512, 4, 33])
        x = self.layer6(x)
        x = self.conv6(x) #torch.Size([256, 512, 4, 33])
        x = self.bn6(x)
        x = self.relu(x)

        x = self.layer7(x)
        x = self.conv7_1(x)
        x = self.bn7_1(x)
        x = self.relu(x)
        x = self.conv7_2(x)
        x = self.bn7_2(x)
        x = self.relu(x)
        return x, FPN