import time
import torch.nn as nn
import torch
import torch.nn.functional as F

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import (
    VGG_FeatureExtractor,
    RCNN_FeatureExtractor,
    ResNet_FeatureExtractor,
)
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention
from modules.Attention_Cls import Attention_Cls
from modules.predicition_2d import Attention as Attention_2D
from modules.Fusion_Seq_Attn_Predicition import Attention_Fusion
from modules.Loss import STR_Loss
from modules.Fusion_Package import GatedBimodal

class One_dimensional_processing_unit(nn.Module):
    def __init__(self, opt):
        super(One_dimensional_processing_unit, self).__init__()
        self.opt = opt
        self.stages = {
            "Trans": opt.Transformation,
            "Feat": opt.FeatureExtraction,
            "Seq": opt.SequenceModeling,
            "Pred": opt.Prediction,
        }

        """ Transformation """
        if opt.Transformation == "TPS":
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial,
                I_size=(opt.imgH, opt.imgW),
                I_r_size=(opt.imgH, opt.imgW),
                I_channel_num=opt.input_channel,
            )
        else:
            print("No Transformation module specified")

        """ freeze TPS weights"""
        for p in self.parameters():
            p.requires_grad = False

        """ FeatureExtraction """
        if opt.FeatureExtraction == "VGG":
            self.FeatureExtraction = VGG_FeatureExtractor(
                opt.input_channel, opt.output_channel
            )
        elif opt.FeatureExtraction == "RCNN":
            self.FeatureExtraction = RCNN_FeatureExtractor(
                opt.input_channel, opt.output_channel
            )
        elif opt.FeatureExtraction == "ResNet":
            self.FeatureExtraction = ResNet_FeatureExtractor(
                opt.input_channel, opt.output_channel
            )
        else:
            raise Exception("No FeatureExtraction module specified")
        self.FeatureExtraction_output = opt.output_channel
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d(
            (None, 1)
        )  # Transform final (imgH/16-1) -> 1

        """Our Sequence modeling"""
        if opt.SequenceModeling == "BiLSTM":
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(
                    self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size
                ),
                BidirectionalLSTM(
                    opt.hidden_size, opt.hidden_size, opt.hidden_size
                ),
            )
            self.SequenceModeling_output = opt.hidden_size
        else:
            print("No SequenceModeling module specified")
            self.SequenceModeling_output = self.FeatureExtraction_output

        self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class)

    def forward(self, image, masks, text=None, is_train=True):
        """Transformation stage"""
        if not self.stages["Trans"] == "None":
            image, masks = self.Transformation(image, masks.float())

        """ Feature extraction stage """
        visual_feature, backbone_feature = self.FeatureExtraction(image)
        visual_feature = visual_feature.permute(0, 3, 1, 2)  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = self.AdaptiveAvgPool(visual_feature)  # [b, w, c, h] -> [b, w, c, 1]
        visual_feature = visual_feature.squeeze(3)  # [b, w, c, 1] -> [b, w, c]

        """ Sequence modeling stage """
        if self.stages["Seq"] == "BiLSTM":
            contextual_feature = self.SequenceModeling(visual_feature)  # [b, num_steps, opt.hidden_size]
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ 1D Decoder"""
        if self.stages["Pred"] == "Attn":
            """ Our """
            pred, seq_attention_map = self.Prediction(contextual_feature.contiguous(), text, is_train,
                                                           batch_max_length=self.opt.batch_max_length)
        return pred, seq_attention_map, backbone_feature, visual_feature, masks


class Two_dimensional_processing_unit(nn.Module):
    def __init__(self, opt):
        super(Two_dimensional_processing_unit, self).__init__()
        self.opt = opt
        self.FeatureExtraction_output = opt.output_channel
        """Prediction"""
        if opt.Prediction == "Attn":
            """TRBA Sequence modeling"""
            if opt.SequenceModeling == "BiLSTM":
                self.SequenceEncoding = nn.Sequential(
                    BidirectionalLSTM(
                        self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size
                    ),
                    BidirectionalLSTM(
                        opt.hidden_size, opt.hidden_size, opt.hidden_size
                    ),
                )
                self.SequenceEncoding_output = opt.hidden_size
            else:
                print("No SequenceModeling module specified")
                self.SequenceEncoding_output = self.FeatureExtraction_output

            self.Prediction = Attention(self.SequenceEncoding_output, opt.hidden_size, opt.num_class)
            self.Attention_Cls = Attention_Cls(opt.batch_max_length + 2)  ###add background layer
            self.SequenceEncoding_output = opt.hidden_size
            self.FC_Prediction_1 = Attention_2D(self.SequenceEncoding_output, opt.hidden_size, opt.num_class)
            self.FC_Prediction_2 = Attention_2D(self.SequenceEncoding_output, opt.hidden_size, opt.num_class)
            self.downsampler = nn.Linear(self.FeatureExtraction_output, opt.hidden_size)

            self.Iterable_Predicition_one = Attention_Fusion(self.SequenceEncoding_output, self.SequenceEncoding_output, opt.num_class)
            self.Iterable_Predicition_two = Attention_Fusion(self.SequenceEncoding_output, self.SequenceEncoding_output, opt.num_class)
            self.Fusion = GatedBimodal(self.SequenceEncoding_output)
            self.pw = nn.Linear(self.SequenceEncoding_output, self.FeatureExtraction_output)
            self.generator = Attention_2D(self.FeatureExtraction_output, self.FeatureExtraction_output, opt.num_class)
        else:
            raise Exception("Prediction is neither CTC or Attn")
        self.loss = STR_Loss()

    def forward(self, backbone_feature, seq_attention_map, visual_feature, masks, length,
                text=None, iteration=None, is_train=True):

        """ Extract 2D Attention maps"""
        char_feature_attn, backfore_feature, Attentive_Sequence = self.Attention_Cls(backbone_feature)

        """ TRBA Modeling 1D Attention maps"""
        contextual_feature = self.SequenceEncoding(visual_feature)
        TRBA_pred, TRBA_attn = self.Prediction(contextual_feature.contiguous(), text, is_train,
                                               batch_max_length=self.opt.batch_max_length)

        """ Loss stage """
        if is_train:
            loss, Single_char_mask, Softmax_classification_Middle, Softmax_classification_Low = \
                self.loss(masks, backfore_feature, seq_attention_map, char_feature_attn,
                          length, iteration, alpha=0.05)
        else:
            loss = None

        Attentive_Sequence_1 = F.dropout(Attentive_Sequence[0], p=0.3, training=is_train)
        pre2D_iter1, _ = self.FC_Prediction_1(Attentive_Sequence_1.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length)
        Iter_pred1, _, _, Iter_fea1 = self.Iterable_Predicition_one(contextual_feature, Attentive_Sequence_1, text, is_train)

        Attentive_Sequence_2 = F.dropout(Attentive_Sequence[1], p=0.3, training=is_train)
        pre2D_iter2, _ = self.FC_Prediction_2(Attentive_Sequence_2.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length)
        Iter_pred2, _, _, Iter_fea2 = self.Iterable_Predicition_two(contextual_feature, Attentive_Sequence_2, text, is_train)

        Enhanced_f = self.Fusion(Iter_fea1, Iter_fea2)
        Complementary_features = self.pw(Enhanced_f)
        Complementary_features = F.dropout(Complementary_features, p=0.3, training=is_train)
        Share_pred, output_hiddens = self.generator(Complementary_features.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length)

        return TRBA_pred, (pre2D_iter1, pre2D_iter2), (Iter_pred1, Iter_pred2), Share_pred, output_hiddens, loss

    def test(self, backbone_feature, visual_feature, text=None, is_train=True):

        """ Extract 2D Attention maps"""
        char_feature_attn, backfore_feature, Attentive_Sequence = self.Attention_Cls(backbone_feature)

        """ TRBA Modeling 1D Attention maps"""
        contextual_feature = self.SequenceEncoding(visual_feature)

        Attentive_Sequence_1 = F.dropout(Attentive_Sequence[0], p=0.3, training=is_train)
        Iter_pred1, _, _, Iter_fea1 = self.Iterable_Predicition_one(contextual_feature, Attentive_Sequence_1, text, is_train)
        Attentive_Sequence_2 = F.dropout(Attentive_Sequence[1], p=0.3, training=is_train)
        Iter_pred2, _, _, Iter_fea2 = self.Iterable_Predicition_two(contextual_feature, Attentive_Sequence_2, text, is_train)

        Enhanced_f = self.Fusion(Iter_fea1, Iter_fea2)
        Complementary_features = self.pw(Enhanced_f)
        Complementary_features = F.dropout(Complementary_features, p=0.3, training=is_train)
        Share_pred, output_hiddens = self.generator(Complementary_features.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length)

        return Share_pred


class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.model_one = One_dimensional_processing_unit(opt)
        self.model_two = Two_dimensional_processing_unit(opt)
    def forward(self, image, masks, length, text=None, iteration=None, is_train=True):
        src_pred, seq_attention_map, backbone_feature, visual_feature, masks = \
            self.model_one(image, masks, text, is_train)
        if is_train:
            TRBA_pred, pre2D_iter, Iter_pred, Share_pred, output_hiddens, loss = \
                self.model_two(backbone_feature, seq_attention_map, visual_feature, masks, length,
                               text, iteration, is_train)
            return (src_pred, TRBA_pred), pre2D_iter, Iter_pred, Share_pred, loss
        else:
            Share_pred = self.model_two.test(backbone_feature, visual_feature, text, is_train)
            return Share_pred
