'''
Implementation of MGP-STR based on ViTSTR.

Copyright 2022 Alibaba
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import logging
import torch.utils.model_zoo as model_zoo

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models import create_model

from modules.segmentor import SegHead
from modules.predicition import Attention, Attention2D, Attention_Fusion, GatedBimodal
from modules.sequential_model import BidirectionalLSTM

_logger = logging.getLogger(__name__)

__all__ = [
    'char_str_base_patch4_3_32_128',
]


def create_char_str(batch_max_length, num_tokens, model=None, checkpoint_path=''):
    char_str = create_model(
        model,
        pretrained=True,
        num_classes=num_tokens,
        checkpoint_path=checkpoint_path,
        batch_max_length=batch_max_length)

    # might need to run to get zero init head for transfer learning
    char_str.reset_classifier(num_classes=num_tokens)

    return char_str


class CHARSTR(VisionTransformer):

    def __init__(self, batch_max_length=26, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.batch_max_length = batch_max_length
        self.out_indices = [2, 4, 6, 8]
        self.norm_seg = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )
        self.segmentation = SegHead(in_channels=self.embed_dim, mla_channels=128, mlahead_channels=128,
                                    num_classes=self.batch_max_length)
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))
        self.SequenceEncoding = nn.Sequential(
            BidirectionalLSTM(512, 256, 256),
            BidirectionalLSTM(256, 256, 256),
        )

    def to_2D(self, x):
        x = x[:, 1:, :]
        return x.reshape(x.shape[0], 8, 32, -1).permute(0, 3, 1, 2)

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.Iterable_Predicition_one = Attention_Fusion(256, 256, num_classes)
        self.Iterable_Predicition_two = Attention_Fusion(256, 256, num_classes)
        self.generator = Attention2D(512, 512, num_classes)
        self.Fusion = GatedBimodal(256)
        self.pw = nn.Linear(256, 512)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # x torch.Size([20, 256, 768])
        x = x + self.pos_embed
        x = self.pos_drop(x)
        Segmentation_input = []
        index = 0
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i + 1 in self.out_indices:
                Segmentation_input.append(self.to_2D(self.norm_seg[index](x)))
                index += 1

        sequential_features, glyph_features, F_score, char_feature_attn = self.segmentation(x, Segmentation_input)
        return sequential_features, glyph_features, F_score, char_feature_attn

    def forward_test_speed(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # x torch.Size([20, 256, 768])
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        Segmentation_input = []
        index = 0
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i + 1 in self.out_indices:
                Segmentation_input.append(self.to_2D(self.norm_seg[index](x)))
                index += 1
        sequential_features, glyph_features, char_feature_attn = self.segmentation.test_speed(x, Segmentation_input)
        return sequential_features, glyph_features, char_feature_attn

    def forward(self, x, mask, text, length, iteration, is_eval=False):
        sequential_features, glyph_features, char_feature_attn = self.forward_test_speed(x)
        visual_feature = sequential_features.permute(0, 3, 1, 2)  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = self.AdaptiveAvgPool(visual_feature)  # [b, w, c, h] -> [b, w, c, 1]
        visual_feature = visual_feature.squeeze(3)  # [b, w, c, 1] -> [b, w, c]
        contextual_feature = self.SequenceEncoding(visual_feature)
        Attentive_Sequence_1 = glyph_features[0]
        Attentive_Sequence_2 = glyph_features[1]
        Iter_pred1, Iter_fea1 = self.Iterable_Predicition_one(contextual_feature.contiguous(), Attentive_Sequence_1,
                                                              text, is_train=not is_eval,
                                                              batch_max_length=self.batch_max_length)
        Iter_pred2, Iter_fea2 = self.Iterable_Predicition_two(contextual_feature.contiguous(), Attentive_Sequence_2,
                                                              text, is_train=not is_eval,
                                                              batch_max_length=self.batch_max_length)
        """ Share Feature """
        Enhanced_f = self.Fusion(Iter_fea1, Iter_fea2)
        Complementary_features = self.pw(Enhanced_f)
        Share_pred, output_hiddens = self.generator(Complementary_features.contiguous(),
                                                    text, is_train=not is_eval, batch_max_length=self.batch_max_length)
        return Share_pred


def load_pretrained(model, cfg=None, num_classes=1000, in_chans=1, filter_fn=None, strict=True):
    '''
    Loads a pretrained checkpoint
    From an older version of timm
    '''
    if cfg is None:
        cfg = getattr(model, 'default_cfg')
    if cfg is None or 'url' not in cfg or not cfg['url']:
        _logger.warning("Pretrained model URL is invalid, using random initialization.")
        return
    torch.hub.load_state_dict_from_url(cfg['url'], progress=True, map_location='cpu')
    state_dict = model_zoo.load_url(cfg['url'], progress=True, map_location='cpu')
    if "model" in state_dict.keys():
        state_dict = state_dict["model"]

    if filter_fn is not None:
        state_dict = filter_fn(state_dict)

    if in_chans == 1:
        conv1_name = cfg['first_conv']
        _logger.info('Converting first conv (%s) pretrained weights from 3 to 1 channel' % conv1_name)
        key = conv1_name + '.weight'
        if key in state_dict.keys():
            _logger.info('(%s) key found in state_dict' % key)
            conv1_weight = state_dict[conv1_name + '.weight']
        else:
            _logger.info('(%s) key NOT found in state_dict' % key)
            return
        # Some weights are in torch.half, ensure it's float for sum on CPU
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        O, I, J, K = conv1_weight.shape
        if I > 3:
            assert conv1_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv1_weight = conv1_weight.reshape(O, I // 3, 3, J, K)
            conv1_weight = conv1_weight.sum(dim=2, keepdim=False)
        else:
            conv1_weight = conv1_weight.sum(dim=1, keepdim=True)
        conv1_weight = conv1_weight.to(conv1_type)
        state_dict[conv1_name + '.weight'] = conv1_weight

    classifier_name = cfg['classifier']
    if num_classes == 1000 and cfg['num_classes'] == 1001:
        # special case for imagenet trained models with extra background class in pretrained weights
        classifier_weight = state_dict[classifier_name + '.weight']
        state_dict[classifier_name + '.weight'] = classifier_weight[1:]
        classifier_bias = state_dict[classifier_name + '.bias']
        state_dict[classifier_name + '.bias'] = classifier_bias[1:]
    elif num_classes != cfg['num_classes']:
        # completely discard fully connected for all other differences between pretrained and created model
        del state_dict[classifier_name + '.weight']
        del state_dict[classifier_name + '.bias']
        strict = False

    print("Loading pre-trained vision transformer weights from %s ..." % cfg['url'])
    model.load_state_dict(state_dict, strict=strict)


def _conv_filter(state_dict):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if not 'patch_embed' in k and not 'pos_embed' in k:
            out_dict[k] = v
        else:
            print("not load", k)
    return out_dict


def char_str_base_patch4_3_32_128(pretrained=False, **kwargs):
    kwargs['in_chans'] = 3
    model = CHARSTR(
        img_size=(32, 128), patch_size=4, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, **kwargs)
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        params_num.append(np.prod(p.size()))
    print(np.sum(params_num))
    return model


model = char_str_base_patch4_3_32_128(False)