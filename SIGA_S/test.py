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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import CharsetMapper
from dataset1 import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model import Model
from Parallel_test import benchmark_all_eval, validate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class init(nn.Module):
    def __init__(self):
        super(init, self).__init__()
        self.dict0 = {"name": 'Share_pred1', "n_correct": 0, "norm_ED": 0, "accuracy": 0.0,
                      "norm_ED_value": 0.0}
        self.dict = [self.dict0
                    ]

def test(opt):
    converter = CharsetMapper(opt.character)
    opt.num_class = len(converter.character)

    model = Model(opt)
    # data parallel for multi-GPU
    model = torch.nn.DataParallel(model).to(device)
    model.eval()
    model.load_state_dict(torch.load(opt.saved_model)['net'])

    """dataset preparation"""
    """
    # opt.eval_type = "benchmark"
    # evaluate = benchmark_all_eval(opt)
    # result_dict = evaluate(model, converter, None, opt, None)
    # print(result_dict)
    """
    opt.select_data = 'ArbitText'
    evaluation_batch_size = opt.batch_size
    eval_data_path = os.path.join(opt.eval_data, opt.select_data)
    AlignCollate_evaluation = AlignCollate(opt, mode="test")
    eval_data, eval_data_log = hierarchical_dataset(root=eval_data_path, opt=opt, mode="test")
    evaluation_loader = torch.utils.data.DataLoader(
        eval_data, batch_size=evaluation_batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_evaluation, pin_memory=True)
    evaluation = validate(opt)
    result_dict, length_of_data = evaluation(model, evaluation_loader, converter, None, opt, tqdm_position=0)
    print(result_dict)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # /home/jcc/GTK/Text Recognition Dataset/training/label
    # /media/xr/guantongkun/downloads/data_lmdb_release/training/label
    parser.add_argument(
        "--eval_data",
        default="../dataset/data_lmdb/evaluation/contextless/",
        help="path to training dataset",
    )
    parser.add_argument("--workers", type=int, default=8, help="number of data loading workers")
    parser.add_argument("--batch_size", type=int, default=64, help="input batch size")
    parser.add_argument("--test_batch_size", type=int, default=64, help="input batch size")
    parser.add_argument("--FT", type=str, default="init", help="whether to do fine-tuning |init|freeze|")
    parser.add_argument("--grad_clip", type=float, default=5, help="gradient clipping value. default=5")
    """ Model Architecture """
    parser.add_argument("--model_name", type=str, required=True, help="CRNN|TRBA")
    parser.add_argument("--num_fiducial", type=int, default=20, help="number of fiducial points of TPS-STN", )
    parser.add_argument("--input_channel", type=int, default=3,
                        help="the number of input channel of Feature extractor", )
    parser.add_argument("--output_channel", type=int, default=384,
                        help="the number of output channel of Feature extractor", )
    parser.add_argument("--hidden_size", type=int, default=256, help="the size of the LSTM hidden state")
    """ Data processing """
    parser.add_argument("--batch_max_length", type=int, default=25, help="maximum-label-length")
    parser.add_argument("--imgH", type=int, default=48, help="the height of the input image")
    parser.add_argument("--imgW", type=int, default=160, help="the width of the input image")
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
    """ Semi-supervised learning """
    parser.add_argument("--semi", type=str, default="None",
                        help="whether to use semi-supervised learning |None|PL|MT|", )
    parser.add_argument("--MT_C", type=float, default=1, help="Mean Teacher consistency weight")
    parser.add_argument("--MT_alpha", type=float, default=0.999, help="Mean Teacher EMA decay")
    parser.add_argument("--model_for_PseudoLabel", default="", help="trained model for PseudoLabel")
    parser.add_argument("--self_pre", type=str, default="RotNet",
                        help="whether to use `RotNet` or `MoCo` pretrained model.", )
    """ exp_name and etc """
    parser.add_argument("--exp_name", help="Where to store logs and models")
    parser.add_argument("--manual_seed", type=int, default=111, help="for random seed setting")
    # ./saved_models/TRBA_synth_SA_/best_accuracy.pth
    parser.add_argument("--saved_model", default="./saved_models/TSBA_synth/best_accuracy.pth", help="path to model to continue training")
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

    elif opt.model_name == "TSBA":  # RBA
        opt.Transformation = "TPS"
        opt.FeatureExtraction = "SVTR"
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
