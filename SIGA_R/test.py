import os
import sys
import time
import random
import string
import argparse
import time
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.init as init
import torch.utils.data
import torch.nn.functional as F
import numpy as np
import re

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import CTCLabelConverter, AttnLabelConverter, Averager, adjust_learning_rate, FCLabelConverter, CharsetMapper
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model import Model
from Parallel_test import benchmark_all_eval
from writer_tensorboard import Writer
from nltk.metrics.distance import edit_distance

import matplotlib.pyplot as plt
from modules.utils import Heatmap
import torchvision.utils as vutils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class init(nn.Module):
    def __init__(self):
        super(init, self).__init__()
        self.dict0 = {"name": 'Share_pred1', "n_correct": 0, "norm_ED": 0, "accuracy": 0.0,
                      "norm_ED_value": 0.0}
        self.dict = [self.dict0]

class validate(nn.Module):
    def __init__(self, opt):
        super(validate, self).__init__()
        self.opt = opt

    def compute_accuracy(self, preds, labels, preds_str, dict, type='Attn'):
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
            if self.opt.Prediction in ['Attn'] and type == 'Attn':
                # if 'Attn' in opt.Prediction:
                pred_EOS = pred.find("[EOS]")
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

            # To evaluate 'case sensitive model' with alphanumeric and case insensitve setting.
            # if self.opt.sensitive:
            pred = pred.lower()
            gt = gt.lower()
            alphanumeric_case_insensitve = '0123456789abcdefghijklmnopqrstuvwxyz'
            out_of_alphanumeric_case_insensitve = f'[^{alphanumeric_case_insensitve}]'
            pred = re.sub(out_of_alphanumeric_case_insensitve, '', pred)
            gt = re.sub(out_of_alphanumeric_case_insensitve, '', gt)

            if pred == gt:
                dict["n_correct"] += 1

            '''
            (old version) ICDAR2017 DOST Normalized Edit Distance https://rrc.cvc.uab.es/?ch=7&com=tasks
            "For each word we calculate the normalized edit distance to the length of the ground truth transcription."
            if len(gt) == 0:
                norm_ED += 1
            else:
                norm_ED += edit_distance(pred, gt) / len(gt)
            '''

            # ICDAR2019 Normalized Edit Distance
            if len(gt) == 0 or len(pred) == 0:
                dict["norm_ED"] += 0
            elif len(gt) > len(pred):
                dict["norm_ED"] += 1 - edit_distance(pred, gt) / len(gt)
            else:
                dict["norm_ED"] += 1 - edit_distance(pred, gt) / len(pred)

    def FC_pred(self, preds, converter):
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index.data)
        return preds_str

    def forward(self, model, evaluation_loader, converter, opt, tqdm_position=1):
        """ validation or evaluation """
        length_of_data = 0
        Init = init()
        for i, (image_tensors, labels, masks) in tqdm(enumerate(evaluation_loader), total=len(evaluation_loader),
                                                      position=tqdm_position, leave=False, ):

            batch_size = image_tensors.size(0)
            length_of_data = length_of_data + batch_size
            image = image_tensors.to(device)
            # For max length prediction
            text_for_pred = (torch.LongTensor(batch_size).fill_(37).to(device))
            length_for_loss = None
            # text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=opt.batch_max_length)

            if 'Attn' in opt.Prediction:
                Share_pred1 = model(image, masks, length_for_loss, text=text_for_pred, is_train=False)
                Str_Share_pred1 = self.FC_pred(Share_pred1, converter)
                self.compute_accuracy(Share_pred1, labels, Str_Share_pred1, Init.dict0, type='FC')

        for dict in Init.dict:
            dict["accuracy"] = dict["n_correct"] / float(length_of_data) * 100
            dict["norm_ED_value"] = dict["norm_ED"] / float(length_of_data)

        return Init.dict


def test(opt):
    converter = CharsetMapper(opt.character)
    opt.num_class = len(converter.character)

    model = Model(opt)
    # data parallel for multi-GPU
    model = torch.nn.DataParallel(model).to(device)
    model.eval()
    pretrained_state_dict = torch.load(opt.saved_model)
    model.load_state_dict(pretrained_state_dict['net'])

    """dataset preparation"""
    evaluation_batch_size = opt.batch_size
    eval_data_path = os.path.join(opt.eval_data, opt.select_data)
    AlignCollate_evaluation = AlignCollate(opt, mode="test")
    eval_data, eval_data_log = hierarchical_dataset(root=eval_data_path, opt=opt, mode="test")
    evaluation_loader = torch.utils.data.DataLoader(
        eval_data, batch_size=evaluation_batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_evaluation, pin_memory=True)
    eval = validate(opt)
    result_dict = eval(model, evaluation_loader, converter, opt, tqdm_position=0)
    print(result_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_data", default="/data/TongkunGuan/data_lmdb_abinet/evaluation/benchmark/",
        help="path to eval dataset",)
    parser.add_argument("--select_data", default="CUTE80", help="path to eval dataset",)
    parser.add_argument("--mask_path", default="", help="path to kmeans results",)
    parser.add_argument("--workers", type=int, default=8, help="number of data loading workers")
    parser.add_argument("--batch_size", type=int, default=1, help="input batch size")
    parser.add_argument("--FT", type=str, default="init", help="whether to do fine-tuning |init|freeze|")
    parser.add_argument("--grad_clip", type=float, default=5, help="gradient clipping value. default=5")
    """ Model Architecture """
    parser.add_argument("--model_name", type=str, default='TRBA', help="CRNN|TRBA")
    parser.add_argument("--num_fiducial", type=int, default=20, help="number of fiducial points of TPS-STN", )
    parser.add_argument("--input_channel", type=int, default=3,
                        help="the number of input channel of Feature extractor", )
    parser.add_argument("--output_channel", type=int, default=512,
                        help="the number of output channel of Feature extractor", )
    parser.add_argument("--hidden_size", type=int, default=256, help="the size of the LSTM hidden state")
    """ Data processing """
    parser.add_argument("--batch_max_length", type=int, default=25, help="maximum-label-length")
    parser.add_argument("--imgH", type=int, default=32, help="the height of the input image")
    parser.add_argument("--imgW", type=int, default=128, help="the width of the input image")
    parser.add_argument(
        "--character",
        type=str,
        # default="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
        default="abcdefghijklmnopqrstuvwxyz1234567890",
        help="character label",
    )
    parser.add_argument("--sensitive", action="store_true", help="For Normalized edit_distance")
    parser.add_argument("--NED", action="store_true", help="For Normalized edit_distance")
    parser.add_argument("--Aug", action="store_true", help="whether to use augmentation |None|Blur|Crop|Rot|", )
    """ exp_name and etc """
    parser.add_argument("--exp_name", default="test", help="Where to store logs and models")
    parser.add_argument("--manual_seed", type=int, default=111, help="for random seed setting")
    parser.add_argument("--saved_model", default="./saved_models/best_score.pth", help="path to model to continue training")
    parser.add_argument("--Iterable_Correct", action="store_true", help="For Normalized edit_distance")
    parser.add_argument("--benchmark_all_eval", action="store_true", help="For Normalized edit_distance")
    parser.add_argument("--language", action="store_true", help="For Normalized edit_distance")

    opt = parser.parse_args()

    if not opt.sensitive:
        opt.character = "abcdefghijklmnopqrstuvwxyz1234567890"
    else:
        opt.character = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

    if opt.model_name == "CRNN":  # CRNN = NVBC
        opt.Transformation = "None"
        opt.FeatureExtraction = "VGG"
        opt.SequenceModeling = "BiLSTM"
        opt.Prediction = "CTC"

    elif opt.model_name == "TRBA":  # TRBA
        opt.Transformation = "TPS"
        opt.FeatureExtraction = "ResNet"
        opt.SequenceModeling = "BiLSTM"
        opt.Prediction = "Attn"

    elif opt.model_name == "RBA":  # RBA
        opt.Transformation = "None"
        opt.FeatureExtraction = "ResNet"
        opt.SequenceModeling = "BiLSTM"
        opt.Prediction = "Attn"

    """ Seed and GPU setting """
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed_all(opt.manual_seed)  # if you are using multi-GPU.
    torch.cuda.manual_seed(opt.manual_seed)

    cudnn.benchmark = True  # It fasten training.
    cudnn.deterministic = True

    opt.gpu_name = "_".join(torch.cuda.get_device_name().split())
    if sys.platform == "linux":
        opt.CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]
    else:
        opt.CUDA_VISIBLE_DEVICES = 0  # for convenience
    test(opt)