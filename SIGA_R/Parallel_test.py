import os
import time
import string
import argparse
import re

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from nltk.metrics.distance import edit_distance

from utils import CTCLabelConverter, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate
from model import Model
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class init(nn.Module):
    def __init__(self):
        super(init, self).__init__()
        self.dict0 = {"name": 'Share_pred', "n_correct": 0, "norm_ED": 0, "accuracy": 0.0,
                      "norm_ED_value": 0.0}
        self.dict1 = {"name": 'language', "n_correct": 0, "norm_ED": 0, "accuracy": 0.0,
                      "norm_ED_value": 0.0}
        self.dict2 = {"name": 'final', "n_correct": 0, "norm_ED": 0, "accuracy": 0.0,
                      "norm_ED_value": 0.0}
        self.dict = [self.dict0, self.dict1, self.dict2]


class validate(nn.Module):
    def __init__(self, opt):
        super(validate, self).__init__()
        self.opt = opt

    def compute_accuracy(self, preds, labels, preds_str, dict, type='Attn'):
        # log = open(f"./saved_models/{self.opt.exp_name}/error.txt", "a")
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
            # else:
            #     log.write(f'{pred}   --->   {gt}')

            '''
            (old version) ICDAR2017 DOST Normalized Edit Distance https://rrc.cvc.uab.es/?ch=7&com=tasks
            "For each word we calculate the normalized edit distance to the length of the ground truth transcription."
            if len(gt) == 0:
                norm_ED += 1
            else:
                norm_ED += edit_distance(pred, gt) / len(gt)
            '''

            # ICDAR2019 Normalized Edit Distance
            # if len(gt) == 0 or len(pred) == 0:
            #     dict["norm_ED"] += 0
            # elif len(gt) > len(pred):
            #     dict["norm_ED"] += 1 - edit_distance(pred, gt) / len(gt)
            # else:
            #     dict["norm_ED"] += 1 - edit_distance(pred, gt) / len(pred)
        # log.close()

    def Attn_pred(self, preds, converter):
        # select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds.max(2)
        preds_size = torch.IntTensor([preds.size(1)] * preds_index.size(0)).to(device)
        preds_str = converter.decode(preds_index, preds_size)
        return preds_str

    def FC_pred(self, preds, converter):
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index.data)
        return preds_str

    def forward(self, model, evaluation_loader, converter, opt, tqdm_position=1):
        """ validation or evaluation """
        length_of_data = 0
        Init = init()

        for i, (image_tensors, labels, masks) in tqdm(enumerate(evaluation_loader)):

            batch_size = image_tensors.size(0)
            length_of_data = length_of_data + batch_size
            image = image_tensors.to(device)
            # For max length prediction
            text_for_pred = (torch.LongTensor(batch_size).fill_(37).to(device))
            text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=opt.batch_max_length)

            if 'Attn' in opt.Prediction:
                Share_pred1 = model(image, masks, length_for_loss, text=text_for_pred, is_train=False)
                Str_Share_pred1 = self.FC_pred(Share_pred1, converter)
                # calculate accuracy & confidence score
                self.compute_accuracy(Share_pred1, labels, Str_Share_pred1, Init.dict0, type='FC')

        for dict in Init.dict:
            dict["accuracy"] = dict["n_correct"] / float(length_of_data) * 100
            dict["norm_ED_value"] = dict["norm_ED"] / float(length_of_data)

        return Init.dict, length_of_data


class benchmark_all_eval(nn.Module):
    def __init__(self, opt):
        super(benchmark_all_eval, self).__init__()
        self.opt = opt
        self.validation = validate(self.opt)
        self.dict = {'Share_pred': 0.0, 'language': 0.0, 'final': 0.0}
        self.method = ['Share_pred', 'language', 'final']

    def forward(self, model, converter, opt, iteration, calculate_infer_time=False):
        """ evaluation with 10 benchmark evaluation datasets """
        if opt.eval_type == "benchmark":
            """evaluation with 6 benchmark evaluation datasets"""
            eval_data_list = [
                "IIIT5k_3000",
                "SVT",
                "IC03_860",
                "IC03_867",
                "IC13_857",
                "IC13_1015",
                "IC15_1811",
                "IC15_2077",
                "SVTP",
                "CUTE80",
            ]
        if calculate_infer_time:
            evaluation_batch_size = 1  # batch_size should be 1 to calculate the GPU inference time per image.
        else:
            evaluation_batch_size = opt.test_batch_size

        total_evaluation_data_number = 0
        total_correct_number = 0
        all_dataset_accuracy = []
        all_dataset_length = []
        evaluation_log = ''
        log = open(f'./saved_models/{opt.exp_name}/result/log_all_evaluation.txt', 'a')
        dashed_line = '-' * 80
        print(dashed_line)
        log.write(dashed_line + '\n')
        evaluation_log += 'iteration: {:} \n'.format(iteration)
        for eval_data_name in eval_data_list:
            eval_data_path = os.path.join(opt.eval_data, eval_data_name)
            AlignCollate_evaluation = AlignCollate(opt, mode="test")
            eval_data, eval_data_log = hierarchical_dataset(root=eval_data_path, opt=opt, mode="test")
            evaluation_loader = torch.utils.data.DataLoader(
                eval_data, batch_size=evaluation_batch_size,
                shuffle=False,
                num_workers=int(opt.workers),
                collate_fn=AlignCollate_evaluation, pin_memory=True)

            result_dict, length_of_data = self.validation(model, evaluation_loader, converter, opt,
                                                          tqdm_position=0)
            total_evaluation_data_number += len(eval_data)
            all_dataset_length.append(length_of_data)
            print(dashed_line)
            evaluation_log += 'dataset: {:} accuracy: '.format(eval_data_name)
            for result in result_dict:
                name = result["name"]
                accuracy = result["accuracy"]
                evaluation_log += f'{name}: {accuracy:0.3f} '
            evaluation_log += '\n'
            all_dataset_accuracy.append(result_dict)

        assert len(self.method) == len(all_dataset_accuracy[0])
        for index_j, name in enumerate(self.method):
            for index_i, result_dict in enumerate(all_dataset_accuracy):
                self.dict[name] += result_dict[index_j]['accuracy'] * all_dataset_length[index_i]
            self.dict[name] /= total_evaluation_data_number
        evaluation_log += 'total_accuracy: '
        for method, accuracy in self.dict.items():
            evaluation_log += f'{method}: {accuracy:0.3f}\t'

        print(evaluation_log)
        log.write(evaluation_log + '\n')
        log.close()

        return self.dict
