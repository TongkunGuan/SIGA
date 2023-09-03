import torch.nn as nn
import torch
import torch.nn.functional as F
from modules.sequence_modeling import BidirectionalLSTM


class GatedBimodal(nn.Module):
    u"""Gated Bimodal neural network.
    Parameters
    ----------
    dim : int
        The dimension of the hidden state.
    activation : :class:`~.bricks.Brick` or None
        The brick to apply as activation. If ``None`` a
        :class:`.Tanh` brick is used.
    gate_activation : :class:`~.bricks.Brick` or None
        The brick to apply as activation for gates. If ``None`` a
        :class:`.Logistic` brick is used.
    Notes
    -----
    See :class:`.Initializable` for initialization parameters.
    """

    def __init__(self, dim, activation=None, gate_activation=None):
        super(GatedBimodal, self).__init__()
        self.dim = dim
        self.W = nn.Linear(2 * dim, dim)

    def forward(self, x_1, x_2):
        if len(x_1.shape) == 2:
            x = torch.cat((x_1, x_2), dim=-1)
            h = F.tanh(x)
            z = F.sigmoid(self.W(x))
            return z * h[:, :self.dim] + (1 - z) * h[:, self.dim:]
        else:
            x = torch.cat((x_1, x_2), dim=-1)
            h = F.tanh(x)
            z = F.sigmoid(self.W(x))
            return z * h[:, :, :self.dim] + (1 - z) * h[:, :, self.dim:]


class SelectiveDecoder(nn.Module):
    def __init__(self, FeatureExtraction_output):
        super(SelectiveDecoder, self).__init__()
        self.FeatureExtraction_output = FeatureExtraction_output
        self.SequenceEncoding = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, self.FeatureExtraction_output,
                              self.FeatureExtraction_output),
            BidirectionalLSTM(self.FeatureExtraction_output, self.FeatureExtraction_output,
                              self.FeatureExtraction_output))
        self.Selective_attention = nn.Linear(FeatureExtraction_output * 2, FeatureExtraction_output)

    def forward(self, contextual_, visual_feature):
        """
        contextual_feature = self.SequenceEncoding(contextual_feature)
        D = torch.cat((contextual_feature, visual_feature), 2)
        attention_map = self.Selective_attention(D)
        x = D * attention_map
        """
        D = torch.cat((contextual_, visual_feature), 2)
        attention_map = torch.sigmoid(self.Selective_attention(D))
        contextual_feature = attention_map * contextual_ + (1 - attention_map) * visual_feature
        contextual_feature = self.SequenceEncoding(contextual_feature)
        return contextual_feature


class Cat_fusion(nn.Module):
    def __init__(self, FeatureExtraction_output):
        super(Cat_fusion, self).__init__()
        self.Cat_head = nn.Linear(FeatureExtraction_output * 2, FeatureExtraction_output)

    def forward(self, contextual_, visual_feature):
        D = torch.cat((contextual_, visual_feature), -1)
        out = self.Cat_head(D)
        return out


class Multi_Modal(nn.Module):
    def __init__(self, FeatureExtraction_output, num_class, num_char_embeddings):
        super(Multi_Modal, self).__init__()
        self.FeatureExtraction_output = FeatureExtraction_output
        # self.char_embeddings = nn.Embedding(num_class, num_char_embeddings)
        self.proj = nn.Linear(num_class, FeatureExtraction_output, False)
        self.Selective_attention_1 = nn.Linear(FeatureExtraction_output * 2, FeatureExtraction_output, bias=True)
        self.Selective_attention_2 = nn.Linear(FeatureExtraction_output * 2, FeatureExtraction_output, bias=True)

        self.Linear1 = nn.Linear(FeatureExtraction_output, FeatureExtraction_output, bias=True)
        self.Linear2 = nn.Linear(FeatureExtraction_output, FeatureExtraction_output, bias=True)

        self.Gate = GatedBimodal(FeatureExtraction_output)

    def forward(self, Iter_fea1, Iter_fea2, Iter_fea3, Iter_pred1):
        preds_prob = torch.softmax(Iter_pred1, dim=2)
        Embedding = self.proj(preds_prob)
        # preds_max_prob, _ = preds_prob.max(dim=2)
        #
        # Embedding = self.char_embeddings(_)

        D1 = torch.cat((Iter_fea2, Embedding), 2)
        attention_map_1 = torch.sigmoid(self.Selective_attention_1(D1))

        D2 = torch.cat((Iter_fea3, Embedding), 2)
        attention_map_2 = torch.sigmoid(self.Selective_attention_2(D2))

        Complementary_features = attention_map_1 * self.Linear1(Iter_fea2) + attention_map_2 * self.Linear2(Iter_fea3)

        Feature = self.Gate(Iter_fea1, Complementary_features)

        return Feature


class Multi_Size(nn.Module):
    def __init__(self):
        super(Multi_Size, self).__init__()

    def forward(self, Iter_fea1, Iter_fea2, Iter_fea3, Iter_pred1, Iter_pred2, Iter_pred3):
        preds_prob1 = torch.softmax(Iter_pred1, dim=2)
        preds_max_prob1, _ = preds_prob1.max(dim=2)
        preds_prob2 = torch.softmax(Iter_pred2, dim=2)
        preds_max_prob2, _ = preds_prob2.max(dim=2)
        preds_prob3 = torch.softmax(Iter_pred3, dim=2)
        preds_max_prob3, _ = preds_prob3.max(dim=2)

        metric = torch.cat(
            [preds_max_prob1.unsqueeze(-1), preds_max_prob2.unsqueeze(-1), preds_max_prob3.unsqueeze(-1)], dim=2)
        metric_ = torch.softmax(metric, dim=2)
        Complementary_features = metric_[:, :, :1] * Iter_fea1 + metric_[:, :, 1:2] * Iter_fea2 + \
                                 metric_[:, :, 2:] * Iter_fea3
        print('current_score:{:.4f}-->{:.4f}-->{:.4f}'.format(metric_[:, :, :1].mean(), metric_[:, :, 1:2].mean(),
                                                              metric_[:, :, 2:].mean()))
        return Complementary_features
