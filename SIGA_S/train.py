import os
import sys
import time
import random
import string
import argparse
import time
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.init as init
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import CTCLabelConverter, AttnLabelConverter, Averager, adjust_learning_rate, FCLabelConverter, CharsetMapper
from dataset1 import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from modules.Loss import MultiLosses
from model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def train(opt, log):
    """dataset preparation"""
    # train dataset. for convenience
    if opt.select_data == "label":
        select_data = [
            "1.SVT",
            "2.IIIT",
            "3.IC13",
            "4.IC15",
            "5.COCO",
            "6.RCTW17",
            "7.Uber",
            "8.ArT",
            "9.LSVT",
            "10.MLT19",
            "11.ReCTS",
        ]

    elif opt.select_data == "synth":
        select_data = ["MJ", "ST"]

    elif opt.select_data == "synth_SA":
        select_data = ["MJ", "ST", "SA"]
        opt.batch_ratio = "0.4-0.4-0.2"  # same ratio with SCATTER paper.

    elif opt.select_data == "mix":
        select_data = [
            "1.SVT",
            "2.IIIT",
            "3.IC13",
            "4.IC15",
            "5.COCO",
            "6.RCTW17",
            "7.Uber",
            "8.ArT",
            "9.LSVT",
            "10.MLT19",
            "11.ReCTS",
            "MJ",
            "ST",
        ]

    elif opt.select_data == "mix_SA":
        select_data = [
            "1.SVT",
            "2.IIIT",
            "3.IC13",
            "4.IC15",
            "5.COCO",
            "6.RCTW17",
            "7.Uber",
            "8.ArT",
            "9.LSVT",
            "10.MLT19",
            "11.ReCTS",
            "MJ",
            "ST",
            "SA",
        ]

    else:
        select_data = opt.select_data.split("-")

    # set batch_ratio for each data.
    if opt.batch_ratio:
        batch_ratio = opt.batch_ratio.split("-")
    else:
        batch_ratio = [round(1 / len(select_data), 3)] * len(select_data)

    train_loader = Batch_Balanced_Dataset(
        opt, opt.train_data, select_data, batch_ratio, log
    )

    print("-" * 80)
    log.write("-" * 80 + "\n")

    """ model configuration """
    if "CTC" in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter_FC = AttnLabelConverter(opt.character)
        converter = CharsetMapper(opt.character)
    opt.num_class = len(converter.character)

    model = Model(opt)

    # weight initialization
    if opt.saved_model == "":
        for name, param in model.named_parameters():
            if "localization_fc2" in name:
                print(f"Skip {name} as it is already initialized")
                continue
            try:
                if "bias" in name:
                    init.constant_(param, 0.0)
                elif "weight" in name:
                    init.kaiming_normal_(param)
            except Exception as e:  # for batchnorm.
                if "weight" in name:
                    param.data.fill_(1)
                continue

    # data parallel for multi-GPU
    model = torch.nn.DataParallel(model).to(device)
    model.train()
    if opt.retrain:
        pretrained_state_dict = torch.load(opt.saved_model)
        model.load_state_dict(torch.load(opt.saved_model)['net'])
    else:
        if opt.saved_model != "":
            print(f"### loading pretrained model from {opt.saved_model}\n")
            pretrained_state_dict = torch.load(opt.saved_model)
            try:
                model.load_state_dict(torch.load(opt.saved_model)['net'])
            except:
                dd = model.state_dict()
                d = pretrained_state_dict['net']
                for i, k in enumerate(d.keys()):
                    try:
                        dd[k] = d[k]
                    except:
                        print(k)
                model.load_state_dict(dd)
        else:
            """ load TPS weights"""
            TPS = torch.load('../SVTR_R/TPS.pth', map_location='cpu')
            IncompatibleKeys = model.load_state_dict(TPS, strict=False)

    log.write(repr(model) + "\n")
    """ setup loss """
    if "CTC" in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = MultiLosses()

    # loss averager
    train_loss_avg = Averager()

    # filter that only require gradient descent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.module.model_one.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    for p in filter(lambda p: p.requires_grad, model.module.model_two.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))

    # setup optimizer
    if opt.optimizer == "adam":
        optimizer = torch.optim.Adam(filtered_parameters, lr=opt.lr)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[67000, 130000], gamma=0.1)
        cycle_momentum = False
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=opt.lr,
            cycle_momentum=cycle_momentum,
            div_factor=20,
            final_div_factor=1000,
            total_steps=int(opt.num_iter * opt.data_ratio),
        )

    elif opt.optimizer == "adamw":
        opt.epoch_num = len(train_loader.dataloader)
        opt.num_iter = 10 * opt.epoch_num
        # no_weight_decay_param_name_list = [n for n, p in model.named_parameters() if any(nd in n for nd in ['norm', 'pos_embed'])]
        # no_weight_decay_param_name_list += [n for n, p in model.named_parameters() if len(p.shape) == 1]
        optimizer = torch.optim.AdamW(params=filtered_parameters, lr=opt.lr, betas=[0.9,0.99], eps=0.00000008, weight_decay=0.05,)
        scheduler = cosine_scheduler(base_value=opt.lr, final_value=0.0000001, epochs=10, niter_per_ep=opt.epoch_num, warmup_epochs=2)

    """ final options """
    # print(opt)
    opt_log = "------------ Options -------------\n"
    args = vars(opt)
    for k, v in args.items():
        if str(k) == "character" and len(str(v)) > 500:
            opt_log += f"{str(k)}: So many characters to show all: number of characters: {len(str(v))}\n"
        else:
            opt_log += f"{str(k)}: {str(v)}\n"
    opt_log += "---------------------------------------\n"
    print(opt_log)
    log.write(opt_log)
    log.close()

    """ start training """
    start_iter = 0
    if opt.retrain:
        try:
            optimizer.load_state_dict(pretrained_state_dict["optimizer"])
            # start_iter = 50000
            start_iter = pretrained_state_dict["iteration"]
            scheduler.last_epoch = start_iter
            scheduler._step_count = start_iter
            # start_iter = int(opt.saved_model.split("_")[-1].split(".")[0])
            print(f"continue to train, start_iter: {start_iter}")
        except:
            pass
    else:
        if opt.saved_model != "":
            try:
                optimizer.load_state_dict(pretrained_state_dict["optimizer"])
                start_iter = pretrained_state_dict["iteration"]
                # scheduler.last_epoch = start_iter
                # scheduler._step_count = start_iter
                # start_iter = int(opt.saved_model.split("_")[-1].split(".")[0])
                print(f"continue to train, start_iter: {start_iter}")
            except:
                pass

    # training loop
    for iteration in tqdm(
            range(start_iter, int(opt.num_iter * opt.data_ratio) + 1),
            total=int(opt.num_iter * opt.data_ratio),
            position=0,
            leave=True,
    ):
        ratio_iteration = int(iteration / opt.data_ratio)
        image_tensors, labels, masks = train_loader.get_batch()

        image = image_tensors.to(device)
        labels_index, labels_length = converter.encode(labels, batch_max_length=opt.batch_max_length + 1)
        batch_size = image.size(0)

        # default recognition loss part
        preds, pre2D, Iter_pred, Share_pred, cost, Map = \
            model(image, masks, labels_length, text=labels_index[:, :-1], iteration=ratio_iteration)  # align with Attention.forward

        target = labels_index[:, 1:]  # without [SOS] Symbol
        loss1D_task = criterion(preds[0], target, labels_length)
        loss1D_1D = criterion(preds[1], target, labels_length)
        loss2D_1 = criterion(pre2D[0], target, labels_length)
        loss2D_2 = criterion(pre2D[1], target, labels_length)
        lossr1 = criterion(Iter_pred[0], target, labels_length)
        lossr2 = criterion(Iter_pred[1], target, labels_length)
        s_lossr1 = criterion(Share_pred, target, labels_length)
        if cost is not None:
            predmask_loss = cost[0].mean()
            Correct_loss = cost[1].mean()
            loss_low, loss_middle = cost[2]
            loss_low = loss_low.mean()
            loss_middle = loss_middle.mean()
        loss = loss1D_task + loss1D_1D + loss2D_1 + loss2D_2 + predmask_loss + \
               0.1*Correct_loss + loss_low + loss_middle + lossr1 + lossr2 + s_lossr1

        if ratio_iteration % opt.log_interval == 0 or ratio_iteration == 1:
            lr = optimizer.param_groups[0]["lr"]
            opt.writer.add_scalar("lr", float(f"{lr:0.7f}"), ratio_iteration)

        model.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_avg.add(loss)

        if "super" in opt.schedule:
            # scheduler.step()
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = scheduler[iteration]
        else:
            adjust_learning_rate(optimizer, iteration, opt)


        if ratio_iteration % 20000 == 0:
            checkpoint = {
                'net': model.state_dict(),
                "optimizer": optimizer.state_dict(),
                'iteration': iteration,
            }
            torch.save(checkpoint, f'./saved_models/{opt.exp_name}/{ratio_iteration}.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data",
        default="../dataset/data_lmdb/training/label/Synth",
        help="path to training dataset",
    )
    parser.add_argument("--workers", type=int, default=12, help="number of data loading workers")
    parser.add_argument("--batch_size", type=int, default=288, help="input batch size")
    parser.add_argument("--test_batch_size", type=int, default=192, help="input batch size")
    parser.add_argument("--data_ratio", type=int, default=1, help="input batch size")
    parser.add_argument("--num_iter", type=int, default=550000, help="number of iterations to train for")
    parser.add_argument("--val_interval", type=int, default=2000, help="Interval between each validation", )
    parser.add_argument("--log_interval", type=int, default=100, help="Interval between each validation", )
    parser.add_argument("--log_multiple_test", action="store_true", help="log_multiple_test")
    parser.add_argument("--FT", type=str, default="init", help="whether to do fine-tuning |init|freeze|")
    parser.add_argument("--grad_clip", type=float, default=None, help="gradient clipping value. default=5")
    """ Optimizer """
    parser.add_argument("--optimizer", type=str, default="adamw", help="optimizer |sgd|adadelta|adam|")
    parser.add_argument("--lr", type=float, default=0.0005,
                        help="learning rate, default=1.0 for Adadelta, 0.0005 for Adam", )
    parser.add_argument("--lr_adadelta", type=float, default=1.0,
                        help="learning rate, default=1.0 for Adadelta, 0.0005 for Adam", )
    parser.add_argument("--sgd_momentum", default=0.9, type=float, help="momentum for SGD")
    parser.add_argument("--sgd_weight_decay", default=0.000001, type=float, help="weight decay for SGD")
    parser.add_argument("--rho", type=float, default=0.95, help="decay rate rho for Adadelta. default=0.95", )
    parser.add_argument("--eps", type=float, default=1e-8, help="eps for Adadelta. default=1e-8")
    parser.add_argument(
        "--schedule",
        default="super",
        nargs="*",
        help="(learning rate schedule. default is super for super convergence, 1 for None, [0.6, 0.8] for the same setting with ASTER",
    )
    parser.add_argument("--lr_drop_rate", type=float, default=0.1,
                        help="lr_drop_rate. default is the same setting with ASTER", )
    """ Model Architecture """
    parser.add_argument("--model_name", type=str, required=True, help="CRNN|TRBA")
    parser.add_argument("--num_fiducial", type=int, default=20, help="number of fiducial points of TPS-STN", )
    parser.add_argument("--input_channel", type=int, default=3,
                        help="the number of input channel of Feature extractor", )
    parser.add_argument("--output_channel", type=int, default=384,
                        help="the number of output channel of Feature extractor", )
    parser.add_argument("--hidden_size", type=int, default=256, help="the size of the LSTM hidden state")
    """ Data processing """
    parser.add_argument("--select_data", type=str, default="label",
                        help="select training data. default is `label` which means 11 real labeled datasets", )
    parser.add_argument("--batch_ratio", type=str, help="assign ratio for each selected data in the batch", )
    parser.add_argument("--total_data_usage_ratio", type=str, default="1.0",
                        help="total data usage ratio, this ratio is multiplied to total number of data.", )
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
    parser.add_argument("--saved_model", default="./saved_models/TSBA_synth/best_accuracy.pth", help="path to model to continue training")
    parser.add_argument("--Iterable_Correct", action="store_true", help="For Normalized edit_distance")
    parser.add_argument("--benchmark_all_eval", action="store_true", help="For Normalized edit_distance")
    parser.add_argument("--saved_model_language", default="", help="path to model to continue training")
    parser.add_argument("--retrain", default=False)
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
    opt.num_gpu = torch.cuda.device_count()
    if opt.num_gpu > 1:
        print(
            "We recommend to use 1 GPU, check your GPU number, you would miss CUDA_VISIBLE_DEVICES=0 or typo"
        )
        print("To use multi-gpu setting, remove or comment out these lines")
        # sys.exit()

    if sys.platform == "win32":
        opt.workers = 0

    """ directory and log setting """
    if not opt.exp_name:
        opt.exp_name = f"Seed{opt.manual_seed}-{opt.model_name}"

    os.makedirs(f"./saved_models/{opt.exp_name}", exist_ok=True)
    os.makedirs(f"./saved_models/{opt.exp_name}/result", exist_ok=True)
    log = open(f"./saved_models/{opt.exp_name}/log_train.txt", "a")
    command_line_input = " ".join(sys.argv)
    print(
        f"Command line input: CUDA_VISIBLE_DEVICES={opt.CUDA_VISIBLE_DEVICES} python {command_line_input}"
    )
    log.write(
        f"Command line input: CUDA_VISIBLE_DEVICES={opt.CUDA_VISIBLE_DEVICES} python {command_line_input}\n"
    )
    os.makedirs(f"./tensorboard", exist_ok=True)
    opt.writer = SummaryWriter(log_dir=f"./tensorboard/{opt.exp_name}")

    train(opt, log)
