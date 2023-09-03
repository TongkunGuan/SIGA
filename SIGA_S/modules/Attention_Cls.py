import torch.nn as nn
import torch
import torch.nn.functional as F
from modules.sequence_modeling import BidirectionalLSTM

class Attention_Cls(nn.Module):
    def __init__(self, batch_max_length=26):
        super(Attention_Cls, self).__init__()
        self.batch_max_length = batch_max_length
        self.unpool2 = nn.Sequential(nn.ConvTranspose2d(128, 64, (4, 3), (2, 1), (1, 1)),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(True))
        self.unpool3 = nn.Sequential(nn.ConvTranspose2d(128, 96, (4, 4), (2, 2), (1, 1)),
                                     nn.BatchNorm2d(96),
                                     nn.ReLU(True))
        self.unpool4 = nn.Sequential(nn.ConvTranspose2d(128, 96, (4, 4), (2, 2), (1, 1)),
                                     nn.BatchNorm2d(96),
                                     nn.ReLU(True))
        self.conv1 = nn.Conv2d(256, 128, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 128, 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(256, 128, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 128, 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(192, 128, 1)
        self.bn5 = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU()
        self.seg_cls = nn.Conv2d(96, 2, 1)
        self.cls_Head_Middle = nn.Sequential(nn.Conv2d(128, batch_max_length, 1, 1, 0))
        self.cls_Head_Low = nn.Sequential(nn.Conv2d(128, batch_max_length, 1, 1, 0))

        self.Middle = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.Low = nn.Sequential(
            nn.Conv2d(192, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.lstm = nn.Sequential(
            BidirectionalLSTM(256, 256, 256),
            BidirectionalLSTM(256, 256, 256),
        )

    def Unet(self, backbone_feature):
        f = backbone_feature
        c = f[2]  # bs 256 6,40
        c = self.conv1(c)
        c = self.bn1(c)
        c = self.relu1(c)

        h = self.conv2(c)
        h = self.bn2(h)
        h = self.relu2(h)
        Middle_cls = self.cls_Head_Middle(h)
        g = self.unpool2(h)  # bs 192 12,40
        c = self.conv3(torch.cat((g, f[1]), 1))
        c = self.bn3(c)
        c = self.relu3(c)

        h = self.conv4(c)
        h = self.bn4(h)
        h = self.relu4(h)
        Low_cls = self.cls_Head_Low(h)
        g = self.unpool3(h)  # bs 96 24,80
        c = self.conv5(torch.cat((g, f[0]), 1))
        c = self.bn5(c)
        c = self.relu5(c)
        c = self.unpool4(c)  # bs 128 48,160
        F_score = self.seg_cls(c)

        return F_score, (Middle_cls, Low_cls)

    def Extract_Feature(self, Attention, Feature):
        nB = Feature.size()[0]
        nC = Feature.size()[1]
        output = Feature.permute(0, 2, 3, 1).view(nB, -1, nC)
        Local_ = torch.bmm(Attention, output)
        return Local_

    def forward(self, backbone_feature):
        backfore_feature, char_feature_attn = self.Unet(backbone_feature)
        Middle_cls, Low_cls = char_feature_attn
        nB, nT = Low_cls.size(0), Low_cls.size(1)
        Low_Feature = self.Low(backbone_feature[1])
        Middle_Feature = self.Middle(backbone_feature[2])

        Local_Low = self.Extract_Feature(torch.softmax(Low_cls.view(nB, nT, -1), dim=-1)[:, 1:, :], Low_Feature)
        Local_Mid = self.Extract_Feature(torch.softmax(Middle_cls.view(nB, nT, -1), dim=-1)[:, 1:, :], Middle_Feature)
        Local_Low = self.lstm[0](Local_Low)
        Local_Mid = self.lstm[1](Local_Mid)
        return char_feature_attn, backfore_feature, (Local_Low, Local_Mid)

