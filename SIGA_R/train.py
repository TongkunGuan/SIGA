import os
import sys
import time
import random
import string
import argparse
import time
import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.init as init
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import utils
from utils import CTCLabelConverter, AttnLabelConverter, Averager, adjust_learning_rate, FCLabelConverter, CharsetMapper
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from modules.Loss import MultiLosses
from model import Model
# from test import validation, benchmark_all_eval
from Parallel_test import benchmark_all_eval, validate
from writer_tensorboard import Writer

import matplotlib.pyplot as plt
from modules.utils import Heatmap
import torchvision.utils as vutils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(opt, log):
    """parameter configuration"""
    utils.init_distributed_mode(opt)

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
    elif opt.select_data == "SD":  # single Dataset
        select_data = ['train']
    else:
        select_data = opt.select_data.split("-")

    # set batch_ratio for each data.
    if opt.batch_ratio:
        batch_ratio = opt.batch_ratio.split("-")
    else:
        batch_ratio = [round(1 / len(select_data), 3)] * len(select_data)

    # multiple datasets
    train_loader = Batch_Balanced_Dataset(
        opt, opt.train_data, select_data, batch_ratio, log
    )
    ''' 
    # single dataset
    AlignCollate_train = AlignCollate(opt, mode="train")
    train_data, train_data_log = hierarchical_dataset(root=opt.train_data, opt=opt, mode="train")
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=opt.batch_size,
        shuffle=True,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_train, pin_memory=True)
    '''
    AlignCollate_evaluation = AlignCollate(opt, mode="test")
    eval_data, eval_data_log = hierarchical_dataset(root=opt.eval_data, opt=opt, mode="test")
    evaluation_loader = torch.utils.data.DataLoader(
        eval_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_evaluation, pin_memory=True)

    print("-" * 80)
    log.write("-" * 80 + "\n")

    """ model configuration """
    if "CTC" in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = CharsetMapper(opt.character)
    opt.num_class = len(converter.character)

    model = Model(opt)

    """ weight initialization """
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

    """ data parallel for multi-GPU """
    if opt.resume:
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
            TPS = torch.load('./TPS.pth', map_location='cpu')
            IncompatibleKeys = model.load_state_dict(TPS, strict=False)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[opt.gpu], find_unused_parameters=True)
    # model = torch.nn.DataParallel(model).to(device)
    model.train()

    log.write(repr(model) + "\n")

    """ setup loss """
    criterion = MultiLosses()

    """ loss averager """
    train_loss_avg = Averager()

    """ filter that only require gradient descent """
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.module.model_one.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    for p in filter(lambda p: p.requires_grad, model.module.model_two.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))

    """ setup optimizer """
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
    if opt.resume:
        try:
            optimizer.load_state_dict(pretrained_state_dict["optimizer"])
            start_iter = pretrained_state_dict["iteration"]
            scheduler.last_epoch = start_iter
            scheduler._step_count = start_iter
            print(f"continue to train, start_iter: {start_iter}")
        except:
            pass
    else:
        if opt.saved_model != "":
            try:
                optimizer.load_state_dict(pretrained_state_dict["optimizer"])
                start_iter = pretrained_state_dict["iteration"]
                scheduler.last_epoch = start_iter
                scheduler._step_count = start_iter
                print(f"continue to train, start_iter: {start_iter}")
            except:
                pass

    start_time = time.time()
    best_accuracy = -1
    best_accuracy_v = -1
    writer = Writer(opt)
    # training loop
    for iteration in tqdm(
            range(start_iter, int(opt.num_iter * opt.data_ratio) + 1),
            total=int(opt.num_iter * opt.data_ratio),
            position=0,
            leave=True,
    ):
        ratio_iteration = int(iteration / opt.data_ratio)
        image_tensors, labels, masks = train_loader.get_batch(ratio_iteration)
        image = image_tensors.to(device)
        labels_index, labels_length = converter.encode(labels, batch_max_length=opt.batch_max_length + 1)
        batch_size = image.size(0)

        # default recognition loss part
        preds, pre2D, Iter_pred, Share_pred, cost = \
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
        if iteration < 20000:
            coefficient = 1.
        else:
            coefficient = 0.3
        loss = loss1D_task + loss1D_1D + loss2D_1 + loss2D_2 + predmask_loss + \
               coefficient * Correct_loss + loss_low + loss_middle + lossr1 + lossr2 + s_lossr1  # <20000 1.0 >20000 0.3 language 0.15

        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
        optimizer.step()
        train_loss_avg.add(loss)

        if "super" in opt.schedule:
            scheduler.step()
        else:
            adjust_learning_rate(optimizer, iteration, opt)

        if ratio_iteration % opt.log_interval == 0 or ratio_iteration == 1 and utils.is_main_process():
            lr = optimizer.param_groups[0]["lr"]
            opt.writer.add_scalar("lr", float(f"{lr:0.7f}"), ratio_iteration)

        torch.cuda.synchronize()

        ###evaluate part
        if (ratio_iteration+1) % opt.val_interval == 0 and utils.is_main_process():  # To see training progress, we also conduct validation when 'iteration == 0'
            elapsed_time = time.time() - start_time
            opt.eval_type = "benchmark"
            model.eval()
            with torch.no_grad():
                if opt.benchmark_all_eval:  # evaluation with 10 benchmark evaluation datasets
                    Evaluation = benchmark_all_eval(opt)
                    Accuracy = Evaluation(model, converter, opt, ratio_iteration)
                    opt.writer.add_scalars('total_accuracy', Accuracy, ratio_iteration)
                    # keep best accuracy model (on valid dataset)
                    for name, accuracy in Accuracy.items():
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            checkpoint = {
                                'net': model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                'iteration': iteration,
                            }
                            torch.save(checkpoint, f'./saved_models/{opt.exp_name}/best_accuracy.pth')
                else:
                    Evaluation = validate(opt)
                    Accuracy = Evaluation(model, evaluation_loader, converter, opt)
                    opt.writer.add_scalar('total_accuracy', Accuracy[0][-3]['accuracy'], ratio_iteration)
                    print(f"accuracy-->{Accuracy[0][-3]['accuracy']}")
                    if Accuracy[0][-3]['accuracy'] > best_accuracy:
                        best_accuracy = Accuracy[0][-3]['accuracy']
                        checkpoint = {
                            'net': model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            'iteration': iteration,
                        }
                        torch.save(checkpoint, f'./saved_models/{opt.exp_name}/best_accuracy.pth')
            print('Iteration:{:}->time:{:}'.format(ratio_iteration, elapsed_time))
            model.train()
        if ratio_iteration % 20000 == 0 and utils.is_main_process():
            checkpoint = {
                'net': model.state_dict(),
                "optimizer": optimizer.state_dict(),
                'iteration': iteration,
            }
            torch.save(checkpoint, f'./saved_models/{opt.exp_name}/{ratio_iteration}.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", default="/data/xxx/data_lmdb_abinet/training/label/Synth/", help="path to training dataset",)
    parser.add_argument("--eval_data", default="/data/xxx/data_lmdb_abinet/evaluation/benchmark/", help="path to eval dataset",)
    parser.add_argument("--mask_path", default='/data/xxx/data_lmdb_abinet/Mask1/',
                        help="path to mask path, optional", )
    parser.add_argument("--workers", type=int, default=8, help="number of data loading workers")
    parser.add_argument("--batch_size", type=int, default=384, help="input batch size")
    parser.add_argument("--test_batch_size", type=int, default=256, help="input batch size")
    parser.add_argument("--data_ratio", type=int, default=1, help="input batch size")
    parser.add_argument("--num_iter", type=int, default=300000, help="number of iterations to train for")
    parser.add_argument("--val_interval", type=int, default=1000, help="Interval between each validation", )
    parser.add_argument("--log_interval", type=int, default=500, help="Interval between each validation", )
    parser.add_argument("--log_multiple_test", action="store_true", help="log_multiple_test")
    parser.add_argument("--FT", type=str, default="init", help="whether to do fine-tuning |init|freeze|")
    parser.add_argument("--grad_clip", type=float, default=5, help="gradient clipping value. default=5")

    """ Optimizer """
    parser.add_argument("--optimizer", type=str, default="adam", help="optimizer |sgd|adadelta|adam|")
    parser.add_argument("--lr", type=float, default=0.00045,
                        help="learning rate, default=1.0 for Adadelta, 0.0005 for Adam", )
    parser.add_argument("--lr_adadelta", type=float, default=1.0,
                        help="learning rate, default=1.0 for Adadelta, 0.0005 for Adam", )
    parser.add_argument("--sgd_momentum", default=0.9, type=float, help="momentum for SGD")
    parser.add_argument("--sgd_weight_decay", default=0.000001, type=float, help="weight decay for SGD")
    parser.add_argument("--rho", type=float, default=0.95, help="decay rate rho for Adadelta. default=0.95", )
    parser.add_argument("--eps", type=float, default=1e-8, help="eps for Adadelta. default=1e-8")
    parser.add_argument("--schedule", default="super", nargs="*",
        help="(learning rate schedule. default is super for super convergence, 1 for None, [0.6, 0.8] for the same setting with ASTER",
    )
    parser.add_argument("--lr_drop_rate", type=float, default=0.1,
                        help="lr_drop_rate. default is the same setting with ASTER", )

    """ Model Architecture """
    parser.add_argument("--model_name", type=str, required=True, help="CRNN|TRBA")
    parser.add_argument("--num_fiducial", type=int, default=20, help="number of fiducial points of TPS-STN", )
    parser.add_argument("--input_channel", type=int, default=3,
                        help="the number of input channel of Feature extractor", )
    parser.add_argument("--output_channel", type=int, default=512,
                        help="the number of output channel of Feature extractor", )
    parser.add_argument("--hidden_size", type=int, default=256, help="the size of the LSTM hidden state")
    """ Data processing """
    parser.add_argument("--select_data", type=str, default="label",
                        help="select training data. default is `label` which means 11 real labeled datasets", )
    parser.add_argument("--batch_ratio", type=str, help="assign ratio for each selected data in the batch", )
    parser.add_argument("--total_data_usage_ratio", type=str, default="1.0",
                        help="total data usage ratio, this ratio is multiplied to total number of data.", )
    parser.add_argument("--batch_max_length", type=int, default=25, help="maximum-label-length")
    parser.add_argument("--imgH", type=int, default=32, help="the height of the input image")
    parser.add_argument("--imgW", type=int, default=128, help="the width of the input image")
    parser.add_argument(
        "--character",
        type=str,
        default="abcdefghijklmnopqrstuvwxyz1234567890",
        help="character label",
    )
    parser.add_argument("--sensitive", action="store_true", help="For Normalized edit_distance")
    parser.add_argument("--NED", action="store_true", help="For Normalized edit_distance")
    parser.add_argument("--Aug", action="store_true", help="whether to use augmentation |None|Blur|Crop|Rot|", )

    """ exp_name and etc """
    parser.add_argument("--exp_name", help="Where to store logs and models")
    parser.add_argument("--manual_seed", type=int, default=111, help="for random seed setting")
    parser.add_argument("--saved_model", default="", help="path to model to continue training")
    parser.add_argument("--benchmark_all_eval", action="store_true", help="For Normalized edit_distance")
    parser.add_argument("--saved_model_language", default="", help="path to model to continue training")
    parser.add_argument("--resume", default=False)

    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
            distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

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
