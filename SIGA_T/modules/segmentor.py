import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .sequential_model import BidirectionalLSTM

class Conv_MLA(nn.Module):
    def __init__(self, in_channels=1024, mla_channels=256):
        super(Conv_MLA, self).__init__()
        self.mla_p2_1x1 = nn.Sequential(nn.Conv2d(
            in_channels, mla_channels, 1, bias=False), nn.BatchNorm2d(mla_channels), nn.ReLU())
        self.mla_p3_1x1 = nn.Sequential(nn.Conv2d(
            in_channels, mla_channels, 1, bias=False), nn.BatchNorm2d(mla_channels), nn.ReLU())
        self.mla_p4_1x1 = nn.Sequential(nn.Conv2d(
            in_channels, mla_channels, 1, bias=False), nn.BatchNorm2d(mla_channels), nn.ReLU())

        self.mla_p2 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1,
                                              bias=False), nn.BatchNorm2d(mla_channels), nn.ReLU())
        self.mla_p3 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1,
                                              bias=False), nn.BatchNorm2d(mla_channels), nn.ReLU())
        self.mla_p4 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1,
                                              bias=False), nn.BatchNorm2d(mla_channels), nn.ReLU())

    def forward(self, res2, res3, res4):
        mla_p4_1x1 = self.mla_p4_1x1(res4)
        mla_p3_1x1 = self.mla_p3_1x1(res3)
        mla_p2_1x1 = self.mla_p2_1x1(res2)

        mla_p3_plus = mla_p4_1x1 + mla_p3_1x1
        mla_p2_plus = mla_p3_plus + mla_p2_1x1

        mla_p4 = self.mla_p4(mla_p4_1x1)
        mla_p3 = self.mla_p3(mla_p3_plus)
        mla_p2 = self.mla_p2(mla_p2_plus)

        return mla_p2, mla_p3, mla_p4

class MLAHead(nn.Module):
    def __init__(self, in_channels=384, mla_channels=128, mlahead_channels=64):
        super(MLAHead, self).__init__()

        self.head2 = nn.Sequential(
            nn.Conv2d(in_channels, mla_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mla_channels),
            nn.ReLU(),
            nn.Conv2d(mla_channels, mlahead_channels, 1, bias=False),
            nn.BatchNorm2d(mlahead_channels),
            nn.ReLU()
          )
        self.head3 = nn.Sequential(
            nn.Conv2d(in_channels, mla_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mla_channels),
            nn.ReLU(),
            nn.Conv2d(mla_channels, mlahead_channels, 1, bias=False),
            nn.BatchNorm2d(mlahead_channels),
            nn.ReLU()
            )

    def forward(self, mla_p2, mla_p3):
        head2 = self.head2(mla_p2)
        head3 = self.head3(mla_p3)
        return torch.cat([head2, head3], dim=1)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

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

class SegHead(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, in_channels=384, mla_channels=128, mlahead_channels=64, num_classes=2, **kwargs):
        super(SegHead, self).__init__(**kwargs)
        batch_max_length = num_classes + 1
        self.token_norm_1 = nn.LayerNorm(in_channels)
        self.token_norm_2 = nn.LayerNorm(in_channels)
        self.inplanes = 512
        self.relu = nn.ReLU(True)
        self.conv5 = nn.Conv2d(in_channels, 512, kernel_size=(2, 3), stride=(2, 1), padding=(0, 1))
        self.layer6 = self._make_layer(BasicBlock, 512, 2, stride=1)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False,)
        self.bn6 = nn.BatchNorm2d(512)
        self.layer7 = self._make_layer(BasicBlock, 512, 1, stride=1)
        self.conv7_1 = nn.Conv2d(512, 512, kernel_size=(2, 3), stride=(2, 1), padding=(0, 1),bias=False,)
        self.bn7_1 = nn.BatchNorm2d(512)
        self.conv7_2 = nn.Conv2d(512, 512, kernel_size=(2, 3), stride=1, padding=(0, 1), bias=False,)
        self.bn7_2 = nn.BatchNorm2d(512)
        self.unpool0 = nn.Sequential(nn.ConvTranspose2d(in_channels, 256, (4, 4), (2, 2), (1, 1)),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(True))

        self.mlahead = MLAHead(in_channels=in_channels, mla_channels=mla_channels, mlahead_channels=mlahead_channels)
        self.cls_Head_middle = nn.Sequential(nn.Conv2d(mlahead_channels * 2, batch_max_length, 1, 1, 0))
        self.cls_Head_low = nn.Sequential(nn.Conv2d(128, batch_max_length, 1, 1, 0))

        self.unpool1 = nn.Sequential(nn.ConvTranspose2d(mlahead_channels * 2, 128, (4, 4), (2, 2), (1, 1)),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(True))
        self.unpool2 = nn.Sequential(nn.ConvTranspose2d(256, 128, (4, 4), (2, 2), (1, 1)),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(True))
        self.unpool3 = nn.Sequential(nn.ConvTranspose2d(128, 128, (4, 4), (2, 2), (1, 1)),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(True))
        self.conv1 = nn.Conv2d(768, 256, 1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(256, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(True)
        self.conv3 = nn.Conv2d(128, 128, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(True)
        self.seg_cls = nn.Conv2d(128, 2, 1)

        self.feat = nn.Conv2d(in_channels, 256, kernel_size = (1,1), stride=1, groups=8, bias=False)

        self.lstm = nn.Sequential(
            BidirectionalLSTM(256, 256, 256),
            BidirectionalLSTM(256, 256, 256),
        )

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

    def Extract_Feature(self, Attention, Feature):
        nB = Feature.size()[0]
        nC = Feature.size()[1]
        output = Feature.permute(0, 2, 3, 1).view(nB, -1, nC)
        Local_ = torch.bmm(Attention, output)
        return Local_


    def forward(self, inputs, segmentation_input):
        ### 1D encoder
        B = inputs.shape[0]
        inputs = self.token_norm_1(inputs) # [bs, 257, 768]
        x = inputs[:, 1:].transpose(1, 2).reshape(B, 768, 8, 32)
        x = self.conv5(x)  # torch.Size([256, 512, 4, 33])
        x = self.layer6(x)
        x = self.conv6(x)  # torch.Size([256, 512, 4, 33])
        x = self.bn6(x)
        x = self.relu(x)
        x = self.layer7(x)
        x = self.conv7_1(x)
        x = self.bn7_1(x)
        x = self.relu(x)
        x = self.conv7_2(x)
        x = self.bn7_2(x)
        sequential_features = self.relu(x)

        x = self.mlahead(segmentation_input[1], segmentation_input[2])
        char_feature_attn_middle = self.cls_Head_middle(x)

        g0 = self.unpool1(x)
        c = self.conv1(segmentation_input[0])
        c = self.bn1(c)
        c = self.relu1(c)
        g1 = self.unpool2(c)
        h = self.conv2(torch.cat((g0, g1), 1))
        h = self.bn2(h)
        h = self.relu2(h)
        char_feature_attn_low = self.cls_Head_low(h)

        g = self.unpool3(h)  # bs 128 16,64
        h = self.conv3(g)
        h = self.bn3(h)
        h = self.relu3(h)
        F_score = self.seg_cls(h)

        nB, nT = char_feature_attn_middle.size(0), char_feature_attn_middle.size(1)
        x = inputs[:, 1:].transpose(1, 2).unsqueeze(-1)  # [bs, 768, 256, 1]
        feat = self.feat(x) #  [bs, 768, 256, 1].
        feat = feat.flatten(2).transpose(1,2)  # [bs, 256, 768]
        selected = torch.softmax(char_feature_attn_middle.view(nB, nT, -1), dim=-1)[:, 1:, :]
        glyph_features_middle = torch.einsum('...si,...id->...sd', selected, feat) # [bs, 26, 768]
        glyph_features_middle = self.lstm[0](glyph_features_middle)

        inputs_high = self.unpool0(segmentation_input[3])
        feat = inputs_high.permute(0, 2, 3, 1).view(nB, -1, 256)
        selected = torch.softmax(char_feature_attn_low.view(nB, nT, -1), dim=-1)[:, 1:, :]
        glyph_features_low = torch.einsum('...si,...id->...sd', selected, feat)  # [bs, 26, 768]
        glyph_features_low = self.lstm[1](glyph_features_low)

        glyph_features = [glyph_features_middle, glyph_features_low]
        char_feature_attn = [char_feature_attn_middle, char_feature_attn_low]

        return sequential_features, glyph_features, F_score, char_feature_attn
    def test_speed(self, inputs, segmentation_input):
        ### 1D encoder
        B = inputs.shape[0]
        inputs = self.token_norm_1(inputs) # [bs, 257, 768]
        x = inputs[:, 1:].transpose(1, 2).reshape(B, 768, 8, 32)
        x = self.conv5(x)  # torch.Size([256, 512, 4, 33])
        x = self.layer6(x)
        x = self.conv6(x)  # torch.Size([256, 512, 4, 33])
        x = self.bn6(x)
        x = self.relu(x)
        x = self.layer7(x)
        x = self.conv7_1(x)
        x = self.bn7_1(x)
        x = self.relu(x)
        x = self.conv7_2(x)
        x = self.bn7_2(x)
        sequential_features = self.relu(x)

        x = self.mlahead(segmentation_input[1], segmentation_input[2])
        char_feature_attn_middle = self.cls_Head_middle(x)

        g0 = self.unpool1(x)
        c = self.conv1(segmentation_input[0])
        c = self.bn1(c)
        c = self.relu1(c)
        g1 = self.unpool2(c)
        h = self.conv2(torch.cat((g0, g1), 1))
        h = self.bn2(h)
        h = self.relu2(h)
        char_feature_attn_low = self.cls_Head_low(h)

        nB, nT = char_feature_attn_middle.size(0), char_feature_attn_middle.size(1)
        x = inputs[:, 1:].transpose(1, 2).unsqueeze(-1)  # [bs, 768, 256, 1]
        feat = self.feat(x) #  [bs, 768, 256, 1].
        feat = feat.flatten(2).transpose(1,2)  # [bs, 256, 768]
        selected = torch.softmax(char_feature_attn_middle.view(nB, nT, -1), dim=-1)[:, 1:, :]
        glyph_features_middle = torch.einsum('...si,...id->...sd', selected, feat) # [bs, 26, 768]
        glyph_features_middle = self.lstm[0](glyph_features_middle)

        inputs_high = self.unpool0(segmentation_input[3])
        feat = inputs_high.permute(0, 2, 3, 1).view(nB, -1, 256)
        selected = torch.softmax(char_feature_attn_low.view(nB, nT, -1), dim=-1)[:, 1:, :]
        glyph_features_low = torch.einsum('...si,...id->...sd', selected, feat)  # [bs, 26, 768]
        glyph_features_low = self.lstm[1](glyph_features_low)

        glyph_features = [glyph_features_middle, glyph_features_low]
        char_feature_attn = [char_feature_attn_middle, char_feature_attn_low]

        return sequential_features, glyph_features, char_feature_attn

