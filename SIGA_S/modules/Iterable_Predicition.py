import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.sequence_modeling import BidirectionalLSTM
from modules.Fusion_Package import GatedBimodal, SelectiveDecoder
from modules.Fusion_Seq_Attn_Predicition import Attention_Fusion
from modules.Fusion_Seq_Predicition import Seq_Fusion


class Iterable_Predicition(nn.Module):
    def __init__(self, opt, input_H, input_W, Correct=False, iterable=1):
        super(Iterable_Predicition, self).__init__()

        self.FeatureExtraction_input = 512
        self.FeatureExtraction_output = 256

        self.Prediction = Attention_Fusion(self.FeatureExtraction_output, self.FeatureExtraction_output, input_H,
                                           input_W, opt.num_class, iterable)
        self.opt = opt
        self.batch_max_length = opt.batch_max_length + 1
        self.batch_size = opt.batch_size
        self.Correct = Correct
        if self.Correct:
            self.dim = 256
            self.SelectiveDecoder = SelectiveDecoder(self.dim)

    def forward(self, Attentive_Sequence, contextual_, text, is_train=True):

        if not self.Correct:
            prediction, seq2_attention_map, output_hiddens, Char = self.Prediction(contextual_.contiguous(),
                                                                                   Attentive_Sequence,
                                                                                   text, is_train,
                                                                                   batch_max_length=self.opt.batch_max_length)
        return prediction, Char, contextual_
