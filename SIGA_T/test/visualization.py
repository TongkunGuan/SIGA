import os
import time
import string
import argparse
import re
import PIL
import validators
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from nltk.metrics.distance import edit_distance

from matplotlib import pyplot as plt
from matplotlib import colors
import cv2
from torchvision import transforms
import torchvision.utils as vutils

from utils import Averager, TokenLabelConverter
from dataset import hierarchical_dataset, AlignCollate, ImgDataset
from models import Model
from utils import get_args
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


def benchmark_all_eval(model, criterion, converter, opt):  # , calculate_infer_time=False):
    """ evaluation with 10 benchmark evaluation datasets """

    if opt.fast_acc:
        # # To easily compute the total accuracy of our paper.
        eval_data_list = ['IC13_857', 'SVT', 'IIIT5k_3000', 'IC15_1811', 'SVTP', 'CUTE80']
    else:
        # The evaluation datasets, dataset order is same with Table 1 in our paper.
        eval_data_list = ['IIIT5k_3000', 'SVT', 'IC03_860', 'IC03_867', 'IC13_857',
                          'IC13_1015', 'IC15_1811', 'IC15_2077', 'SVTP', 'CUTE80']
        # eval_data_list = ['ArbitText']

    if opt.calculate_infer_time:
        evaluation_batch_size = 1  # batch_size should be 1 to calculate the GPU inference time per image.
    else:
        evaluation_batch_size = opt.batch_size

    char_list_accuracy = []
    bpe_list_accuracy = []
    wp_list_accuracy = []
    fused_list_accuracy = []
    total_forward_time = 0
    total_evaluation_data_number = 0
    char_total_correct_number = 0
    bpe_total_correct_number = 0
    wp_total_correct_number = 0
    fused_total_correct_number = 0
    log = open(f'./result/{opt.exp_name}/log_all_evaluation.txt', 'a')
    dashed_line = '-' * 80
    print(dashed_line)
    log.write(dashed_line + '\n')
    for eval_data in eval_data_list:

        if opt.eval_img:
            eval_data_path = os.path.join(opt.eval_data, eval_data + '.txt')
            eval_data = ImgDataset(root=eval_data_path, opt=opt)
        else:
            eval_data_path = os.path.join(opt.eval_data, eval_data)
            print(eval_data_path)
            eval_data, eval_data_log = hierarchical_dataset(root=eval_data_path, opt=opt)

        AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, opt=opt)
        evaluation_loader = torch.utils.data.DataLoader(
            eval_data, batch_size=evaluation_batch_size,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_evaluation, pin_memory=False)

        _, accuracys, _, _, _, infer_time, length_of_data, accur_numbers = validation(
            model, criterion, evaluation_loader, converter, opt)
        char_list_accuracy.append(f'{accuracys[1]:0.3f}')
        bpe_list_accuracy.append(f'{accuracys[2]:0.3f}')
        wp_list_accuracy.append(f'{accuracys[3]:0.3f}')
        fused_list_accuracy.append(f'{accuracys[4]:0.3f}')

        total_forward_time += infer_time
        total_evaluation_data_number += len(eval_data)
        char_total_correct_number += accur_numbers[0]
        bpe_total_correct_number += accur_numbers[1]
        wp_total_correct_number += accur_numbers[2]
        fused_total_correct_number += accur_numbers[4]
        # log.write(eval_data_log)
        print(
            f'char_Acc {accuracys[0]:0.3f}\t bpe_Acc {accuracys[1]:0.3f}\t wp_Acc {accuracys[2]:0.3f}\t  fused_Acc {accuracys[3]:0.3f}')
        log.write(
            f'char_Acc {accuracys[0]:0.3f}\t bpe_Acc {accuracys[1]:0.3f}\t wp_Acc {accuracys[2]:0.3f}\t fused_Acc {accuracys[3]:0.3f}')
        print(dashed_line)
        log.write(dashed_line + '\n')

    averaged_forward_time = total_forward_time / total_evaluation_data_number * 1000
    char_total_accuracy = round(char_total_correct_number / total_evaluation_data_number * 100, 3)
    bpe_total_accuracy = round(bpe_total_correct_number / total_evaluation_data_number * 100, 3)
    wp_total_accuracy = round(wp_total_correct_number / total_evaluation_data_number * 100, 3)
    fused_total_accuracy = round(fused_total_correct_number / total_evaluation_data_number * 100, 3)
    params_num = sum([np.prod(p.size()) for p in model.parameters()])

    evaluation_log = 'accuracy: ' + '\n'
    evaluation_log += 'char_total_Acc:' + str(char_total_accuracy) + '\n' + 'bpe_total_Acc:' + str(
        bpe_total_accuracy) + '\n' + 'wp_total_Acc:' + str(wp_total_accuracy) + '\n' + 'fused_total_Acc:' + str(
        fused_total_accuracy) + '\n'
    evaluation_log += f'averaged_infer_time: {averaged_forward_time:0.3f}\t# parameters: {params_num / 1e6:0.3f}'
    if opt.flops:
        evaluation_log += get_flops(model, opt, converter)
    print(evaluation_log)
    log.write(evaluation_log + '\n')
    log.close()

    return [char_total_accuracy, bpe_total_accuracy, wp_total_accuracy, fused_total_accuracy]


def decode(pred, converter, batch_size):
    _, char_pred_index = pred.topk(1, dim=-1, largest=True, sorted=True)
    char_pred_index = char_pred_index.view(-1, converter.batch_max_length - 1)
    length_for_pred = torch.IntTensor([converter.batch_max_length - 1] * batch_size).to(device)
    char_preds_str = converter.char_decode(char_pred_index, length_for_pred)
    char_pred_prob = F.softmax(pred, dim=2)
    char_pred_max_prob, _ = char_pred_prob.max(dim=2)
    return char_preds_str, char_pred_max_prob


def validation(model, criterion, evaluation_loader, converter, opt):
    """ validation or evaluation """
    for i, (image_tensors, masks, labels, imgs_path) in tqdm(enumerate(evaluation_loader)):
        batch_size = image_tensors.size(0)
        image = image_tensors.to(device)
        # For max length prediction
        if opt.Transformer:
            len_target, char_target = converter.char_encode(labels)
        else:
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
            text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=opt.batch_max_length)

        Ours_pred, TRBA_pred, (pre2D_iter1, pre2D_iter2), (Iter_pred1, Iter_pred2), Share_pred, loss, map = \
            model(image, masks, char_target, len_target, iteration=0, is_eval=False)  # final
        (seq_attention_map, char_feature_attn, Single_char_mask, fore_mask) = map
        try:
            for i in range(batch_size):
                if len_target[i] < 5:
                    continue
                if (len_target[i] - 1) % 2 == 1:
                    num = (len_target[i] - 1) // 2 + 1
                else:
                    num = (len_target[i] - 1) // 2
                if not os.path.exists(f'../demo_imgs/demo5/{num}/'):
                    os.makedirs(f'../demo_imgs/demo5/{num}/')

                # plt.imshow(image_tensors[i].permute(1, 2, 0).cpu().numpy())
                # plt.axis('off')
                # plt.savefig(f'../demo_imgs/demo1/{num}/{labels[i]}_image.jpg', bbox_inches='tight', pad_inches=0)
                # plt.close()
                # plt.imshow(masks[i].squeeze().numpy())
                # plt.axis('off')
                # plt.savefig(f'../demo_imgs/demo1/{num}/{labels[i]}_mask.jpg', bbox_inches='tight', pad_inches=0)
                # plt.close()
                # plt.imshow(fore_mask[i].squeeze().cpu().numpy())
                # plt.axis('off')
                # plt.savefig(f'../demo_imgs/demo1/{num}/{labels[i]}_pred.jpg', bbox_inches='tight', pad_inches=0)
                # plt.close()
                # Attn_ = seq_attention_map[i].permute(1, 0)[:len_target[1] - 1]
                # Attn_ = torch.nn.functional.interpolate(Attn_.unsqueeze(0).unsqueeze(1).float(),
                #                                         size=(len_target[1] - 1, 128),
                #                                         scale_factor=None, mode='bilinear',
                #                                         align_corners=None).squeeze()
                # # Attn_ = Attn_>0.05
                # Attn_ = torch.cat([Attn_, torch.zeros(33 - len_target[1], 128)], dim=0)
                # plt.imshow(Attn_.cpu().numpy())
                # plt.axis('off')
                # plt.savefig(f'../demo_imgs/demo1/{num}/{labels[i]}_seq.jpg', bbox_inches='tight', pad_inches=0)
                # plt.close()
                #
                # grid = vutils.make_grid(Single_char_mask[i][1:len_target[i]].unsqueeze(1).float(), normalize=True, padding=0,
                #                         scale_each=True, nrow=(len_target[i]-1)//2+1)
                # grid = grid.permute(1, 2, 0)
                # plt.imshow(grid.numpy())
                # plt.axis('off')
                # plt.savefig(f'../demo_imgs/demo1/{num}/{labels[i]}_char.jpg', bbox_inches='tight', pad_inches=0)
                # plt.close()
                #
                # char_feature = F.softmax(char_feature_attn[1], dim=1)
                # overlaps = []
                # ABI_scores = char_feature[i][:len_target[i]].reshape(-1, 16, 64)
                # T = ABI_scores.shape[0]
                # attn_scores = ABI_scores.detach().cpu().numpy()
                # image_numpy = image_tensors[i].detach().cpu().float().numpy()
                # if image_numpy.shape[0] == 1:
                #     image_numpy = np.tile(image_numpy, (3, 1, 1))
                # x = (np.transpose(image_numpy, (1, 2, 0))+1)/2. * 255.0
                # for t in range(T):
                #     att_map = attn_scores[t, :, :]  # [feature_H, feature_W, 1] .reshape(32, 8).T
                #     # normalize mask
                #     att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + np.finfo(float).eps)
                #     x = cv2.resize(x, (64, 16))  # [H, W]
                #     # x = cv2.resize(x, (128, 32))
                #     att_map = (att_map * 255).astype(np.uint8)
                #     heatmap = cv2.applyColorMap(att_map, cv2.COLORMAP_JET)  # [H, W, C]
                #     overlap = cv2.addWeighted(heatmap, 0.6, x, 0.4, 0, dtype=cv2.CV_32F)
                #     overlaps.append(overlap)
                # char_segmention = vutils.make_grid(torch.Tensor(overlaps[1:]).permute(0, 3, 1, 2), normalize=True, padding=0,
                #                                    scale_each=True, nrow=(len_target[i]-1)//2+1)
                # char_segmention = char_segmention.permute(1, 2, 0)
                # plt.imshow(char_segmention.numpy())
                # plt.axis('off')
                # plt.savefig(f'../demo_imgs/demo1/{num}/{labels[i]}_attn.jpg', bbox_inches='tight', pad_inches=0)
                # plt.close()
                # 0.7725, 0.3529, 0.06666

                # fill_value = torch.Tensor([0.7686, 0.8, 0.949])
                # mask_pred = torch.cat(
                #     [masks[i].squeeze().unsqueeze(2).repeat(1, 1, 3), fill_value.reshape(1, 1, 3).repeat(3, 128, 1),
                #      fore_mask[i].squeeze().unsqueeze(2).repeat(1, 1, 3)], dim=0)
                # Attn_ = seq_attention_map[i].permute(1, 0)[:len_target[i] - 1]
                # Attn_ = torch.nn.functional.interpolate(Attn_.unsqueeze(0).unsqueeze(1).float(),
                #                                         size=(len_target[i] - 1, 128),
                #                                         scale_factor=None, mode='bilinear',
                #                                         align_corners=None).squeeze()
                # Attn_ = Attn_ > 0.05
                # Attn_ = torch.cat([Attn_, torch.zeros(33 - len_target[i], 128)], dim=0)
                # img_seq = torch.cat([image_tensors[i].permute(1, 2, 0), fill_value.reshape(1, 1, 3).repeat(3, 128, 1),
                #                      Attn_.unsqueeze(-1).repeat(1, 1, 3)], dim=0)
                #
                # grid = vutils.make_grid(Single_char_mask[i][1:len_target[i]].unsqueeze(1).float(), normalize=True,
                #                         padding=0,
                #                         scale_each=True, nrow=(len_target[i] - 1) // 2 + 1)
                # grid = grid.permute(1, 2, 0)
                # char_feature = F.softmax(char_feature_attn[1], dim=1)
                # overlaps = []
                # ABI_scores = char_feature[i][:len_target[i]].reshape(-1, 16, 64)
                # T = ABI_scores.shape[0]
                # attn_scores = ABI_scores.detach().cpu().numpy()
                # image_numpy = image_tensors[i].detach().cpu().float().numpy()
                # if image_numpy.shape[0] == 1:
                #     image_numpy = np.tile(image_numpy, (3, 1, 1))
                # # x = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2. * 255.0
                # x = np.transpose(image_numpy, (1, 2, 0)) * 255.0
                #
                # for t in range(T):
                #     att_map = attn_scores[t, :, :]  # [feature_H, feature_W, 1] .reshape(32, 8).T
                #     # normalize mask
                #     att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + np.finfo(float).eps)
                #     x = cv2.resize(x, (64, 16))  # [H, W]
                #     # x = cv2.resize(x, (128, 32))
                #     att_map = (att_map * 255).astype(np.uint8)
                #     heatmap = cv2.applyColorMap(att_map, cv2.COLORMAP_JET)  # [H, W, C]
                #     heatmap = heatmap[:, :, ::-1]
                #     overlap = cv2.addWeighted(x, 0.5, heatmap, 0.5, 0, dtype=cv2.CV_32F)
                #     overlaps.append(overlap)
                # char_segmention = vutils.make_grid(torch.Tensor(overlaps[1:]).permute(0, 3, 1, 2), normalize=True,
                #                                    padding=0,
                #                                    scale_each=True, nrow=(len_target[i] - 1) // 2 + 1)
                # char_segmention = char_segmention.permute(1, 2, 0)
                # # char_segmention = torch.Tensor(np.array(char_segmention)[:, :, ::-1].copy())
                # char_attn = torch.cat([grid, fill_value.reshape(1, 1, 3).repeat(3, grid.shape[1], 1), char_segmention], dim=0)
                # final_vis = torch.cat([img_seq, fill_value.reshape(1, 1, 3).repeat(67, 3, 1), mask_pred,
                #                        fill_value.reshape(1, 1, 3).repeat(67, 3, 1), char_attn], dim=1)
                # final_vis = torch.cat([fill_value.reshape(1, 1, 3).repeat(3, final_vis.shape[1], 1), final_vis,
                #                        fill_value.reshape(1, 1, 3).repeat(3, final_vis.shape[1], 1)], dim=0)
                # final_vis = torch.cat([fill_value.reshape(1, 1, 3).repeat(final_vis.shape[0], 3, 1), final_vis,
                #                        fill_value.reshape(1, 1, 3).repeat(final_vis.shape[0], 3, 1)], dim=1)
                #
                # plt.imshow(final_vis.numpy())
                # plt.axis('off')
                # plt.savefig(f'../demo_imgs/demo2/{num}/{labels[i]}.jpg', bbox_inches='tight', pad_inches=0, dpi=300)
                # plt.close()

                # fill_value = torch.Tensor([0.7686, 0.8, 0.949])
                # mask_pred = torch.cat(
                #     [masks[i].squeeze().unsqueeze(2).repeat(1, 1, 3), fill_value.reshape(1, 1, 3).repeat(32, 3, 1),
                #      fore_mask[i].squeeze().unsqueeze(2).repeat(1, 1, 3)], dim=1)
                # Attn_ = seq_attention_map[i].permute(1, 0)[:len_target[i] - 1]
                # Attn_ = torch.nn.functional.interpolate(Attn_.unsqueeze(0).unsqueeze(1).float(),
                #                                         size=(len_target[i] - 1, 128),
                #                                         scale_factor=None, mode='bilinear',
                #                                         align_corners=None).squeeze()
                # Attn_ = Attn_ > 0.05
                # Attn_ = torch.cat([Attn_, torch.zeros(33 - len_target[i], 128)], dim=0)
                # img_seq = torch.cat([image_tensors[i].permute(1, 2, 0), fill_value.reshape(1, 1, 3).repeat(32, 3, 1),
                #                      Attn_.unsqueeze(-1).repeat(1, 1, 3)], dim=1)
                #
                # grid = vutils.make_grid(Single_char_mask[i][1:len_target[i]].unsqueeze(1).float(), normalize=True,
                #                         padding=0,
                #                         scale_each=True, nrow=(len_target[i] - 1) // 2 + 1)
                # grid = grid.permute(1, 2, 0)
                # char_feature = F.softmax(char_feature_attn[1], dim=1)
                # overlaps = []
                # ABI_scores = char_feature[i][:len_target[i]].reshape(-1, 16, 64)
                # T = ABI_scores.shape[0]
                # attn_scores = ABI_scores.detach().cpu().numpy()
                # image_numpy = image_tensors[i].detach().cpu().float().numpy()
                # if image_numpy.shape[0] == 1:
                #     image_numpy = np.tile(image_numpy, (3, 1, 1))
                # # x = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2. * 255.0
                # x = np.transpose(image_numpy, (1, 2, 0)) * 255.0
                #
                # for t in range(T):
                #     att_map = attn_scores[t, :, :]  # [feature_H, feature_W, 1] .reshape(32, 8).T
                #     # normalize mask
                #     att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + np.finfo(float).eps)
                #     x = cv2.resize(x, (64, 16))  # [H, W]
                #     # x = cv2.resize(x, (128, 32))
                #     att_map = (att_map * 255).astype(np.uint8)
                #     heatmap = cv2.applyColorMap(att_map, cv2.COLORMAP_JET)  # [H, W, C]
                #     heatmap = heatmap[:, :, ::-1]
                #     overlap = cv2.addWeighted(x, 0.5, heatmap, 0.5, 0, dtype=cv2.CV_32F)
                #     overlaps.append(overlap)
                # char_segmention = vutils.make_grid(torch.Tensor(overlaps[1:]).permute(0, 3, 1, 2), normalize=True,
                #                                    padding=0,
                #                                    scale_each=True, nrow=(len_target[i] - 1) // 2 + 1)
                # char_segmention = char_segmention.permute(1, 2, 0)
                # # char_segmention = torch.Tensor(np.array(char_segmention)[:, :, ::-1].copy())
                # char_attn = torch.cat([grid, fill_value.reshape(1, 1, 3).repeat(32, 3, 1), char_segmention], dim=1)
                # final_vis = torch.cat([img_seq, fill_value.reshape(1, 1, 3).repeat(32, 3, 1), mask_pred,
                #                        fill_value.reshape(1, 1, 3).repeat(32, 3, 1), char_attn], dim=1)
                # final_vis = torch.cat([fill_value.reshape(1, 1, 3).repeat(3, final_vis.shape[1], 1), final_vis,
                #                        fill_value.reshape(1, 1, 3).repeat(3, final_vis.shape[1], 1)], dim=0)
                # final_vis = torch.cat([fill_value.reshape(1, 1, 3).repeat(final_vis.shape[0], 3, 1), final_vis,
                #                        fill_value.reshape(1, 1, 3).repeat(final_vis.shape[0], 3, 1)], dim=1)
                #
                # plt.imshow(final_vis.numpy())
                # plt.axis('off')
                # plt.savefig(f'../demo_imgs/demo3/{num}/{labels[i]}.jpg', bbox_inches='tight', pad_inches=0, dpi=300)
                # plt.close()

                fill_value = torch.Tensor([0.7686, 0.8, 0.949])
                mask_pred = torch.cat(
                    [masks[i].squeeze().unsqueeze(2).repeat(1, 1, 3), fill_value.reshape(1, 1, 3).repeat(4, 128, 1),
                     fore_mask[i].squeeze().unsqueeze(2).repeat(1, 1, 3)], dim=0)
                Attn_ = seq_attention_map[i].permute(1, 0)[:len_target[i] - 1]
                Attn_ = torch.nn.functional.interpolate(Attn_.unsqueeze(0).unsqueeze(1).float(),
                                                        size=(len_target[i] - 1, 128),
                                                        scale_factor=None, mode='bilinear',
                                                        align_corners=None).squeeze()
                Attn_ = Attn_ > 0.05
                Attn_ = torch.cat([Attn_, torch.zeros(33 - len_target[i], 128)], dim=0)
                img_seq = torch.cat([image_tensors[i].permute(1, 2, 0), fill_value.reshape(1, 1, 3).repeat(4, 128, 1),
                                     Attn_.unsqueeze(-1).repeat(1, 1, 3)], dim=0)

                grid = torch.nn.functional.interpolate(Single_char_mask[i].unsqueeze(0).float(), size=(
                    32, 128), scale_factor=None, mode='bilinear', align_corners=None).squeeze()

                if (len_target[i] - 1) % 2 == 1:
                    col_num = (len_target[i] - 1) // 2 + 1
                else:
                    col_num = (len_target[i] - 1) // 2

                grid = grid.unsqueeze(1).float()
                if grid.dim() == 4 and grid.size(1) == 1:  # single-channel images
                    grid = torch.cat((grid, grid, grid), 1)
                grid = grid.permute(0, 2, 3, 1)
                fill_zeros = torch.zeros((32+4+32, 128*col_num+2*(col_num-1), 3))

                last_index = 0
                for j in range(1, col_num + 1):
                    fill_zeros[:32, last_index:last_index + 128, :] = grid[j]
                    last_index += 128
                    try:
                        fill_zeros[:32, last_index:last_index + 2, :] = fill_value.reshape(1, 1, 3).repeat(32, 2, 1)
                        last_index += 2
                    except:
                        state = 0
                fill_zeros[32:33, :, :] = torch.zeros(1, 128 * col_num + 2 * (col_num - 1), 3)
                fill_zeros[33:35, :, :] = fill_value.reshape(1, 1, 3).repeat(2, 128*col_num+2*(col_num-1), 1)
                fill_zeros[35:36, :, :] = torch.zeros(1, 128 * col_num + 2 * (col_num - 1), 3)
                last_index = 0
                for j in range(col_num + 1, len_target[i]):
                    fill_zeros[36:, last_index:last_index + 128, :] = grid[j]
                    last_index += 128
                    try:
                        fill_zeros[36:, last_index:last_index + 2, :] = fill_value.reshape(1, 1, 3).repeat(32, 2, 1)
                        last_index += 2
                    except:
                        state = 0
                grid = fill_zeros

                char_feature = F.softmax(char_feature_attn[1], dim=1)
                overlaps = []
                ABI_scores = char_feature[i][:len_target[i]].reshape(-1, 16, 64)
                T = ABI_scores.shape[0]
                attn_scores = ABI_scores.detach().cpu().numpy()
                image_numpy = image_tensors[i].detach().cpu().float().numpy()
                if image_numpy.shape[0] == 1:
                    image_numpy = np.tile(image_numpy, (3, 1, 1))
                # x = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2. * 255.0
                x = np.transpose(image_numpy, (1, 2, 0)) * 255.0

                for t in range(T):
                    att_map = attn_scores[t, :, :]  # [feature_H, feature_W, 1] .reshape(32, 8).T
                    # normalize mask
                    att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + np.finfo(float).eps)
                    # x = cv2.resize(x, (64, 16))  # [H, W]
                    att_map = cv2.resize(att_map, (128, 32))
                    att_map = (att_map * 255).astype(np.uint8)
                    heatmap = cv2.applyColorMap(att_map, cv2.COLORMAP_JET)  # [H, W, C]
                    heatmap = heatmap[:, :, ::-1]
                    overlap = cv2.addWeighted(x, 0.5, heatmap, 0.5, 0, dtype=cv2.CV_32F)
                    overlaps.append(overlap)
                overlaps = torch.Tensor(overlaps)
                low = overlaps.min()
                high = overlaps.max()
                overlaps.sub_(low).div_(max(high - low, 1e-5))

                fill_zeros = torch.zeros((32 + 4 + 32, 128 * col_num + 2 * (col_num - 1), 3))
                last_index = 0
                for j in range(1, col_num + 1):
                    fill_zeros[:32, last_index:last_index + 128, :] = overlaps[j]
                    last_index += 128
                    try:
                        fill_zeros[:32, last_index:last_index + 2, :] = fill_value.reshape(1, 1, 3).repeat(32, 2, 1)
                        last_index += 2
                    except:
                        state = 0
                fill_zeros[32:33, :, :] = torch.zeros(1, 128 * col_num + 2 * (col_num - 1), 3)
                fill_zeros[33:35, :, :] = fill_value.reshape(1, 1, 3).repeat(2, 128 * col_num + 2 * (col_num - 1), 1)
                fill_zeros[35:36, :, :] = torch.zeros(1, 128 * col_num + 2 * (col_num - 1), 3)
                last_index = 0
                for j in range(col_num + 1, len_target[i]):
                    fill_zeros[36:, last_index:last_index + 128, :] = overlaps[j]
                    last_index += 128
                    try:
                        fill_zeros[36:, last_index:last_index + 2, :] = fill_value.reshape(1, 1, 3).repeat(32, 2, 1)
                        last_index += 2
                    except:
                        state = 0
                char_segmention = fill_zeros


                char_attn = torch.cat([grid, fill_value.reshape(1, 1, 3).repeat(68, 4, 1), char_segmention], dim=1)
                final_vis = torch.cat([img_seq, fill_value.reshape(1, 1, 3).repeat(68, 4, 1), mask_pred,
                                       fill_value.reshape(1, 1, 3).repeat(68, 4, 1), char_attn], dim=1)
                final_vis = torch.cat([fill_value.reshape(1, 1, 3).repeat(4, final_vis.shape[1], 1), final_vis,
                                       fill_value.reshape(1, 1, 3).repeat(4, final_vis.shape[1], 1)], dim=0)
                final_vis = torch.cat([fill_value.reshape(1, 1, 3).repeat(final_vis.shape[0], 4, 1), final_vis,
                                       fill_value.reshape(1, 1, 3).repeat(final_vis.shape[0], 4, 1)], dim=1)

                plt.imshow(final_vis.numpy())
                plt.axis('off')
                plt.savefig(f'../demo_imgs/demo5/{num}/{labels[i]}.jpg', bbox_inches='tight', pad_inches=0, dpi=300)
                plt.close()

        except:
            print()


def draw_atten(img_path, gt, pred, attn, pil, tensor, resize, count, flag=0):
    image = PIL.Image.open(img_path).convert('RGB')
    image = cv2.resize(np.array(image), (128, 32))
    image = tensor(image)
    image_np = np.array(pil(image))

    attn_pil = [pil(a) for a in attn[:, None, :, :]]
    attn = [tensor(resize(a)).repeat(3, 1, 1) for a in attn_pil]
    attn_sum = np.array([np.array(a) for a in attn_pil[:len(pred)]]).sum(axis=0)
    blended_sum = tensor(blend_mask(image_np, attn_sum))
    blended = [tensor(blend_mask(image_np, np.array(a))) for a in attn_pil]
    save_image = torch.stack([image] + attn + [blended_sum] + blended)
    save_image = save_image.view(2, -1, *save_image.shape[1:])
    save_image = save_image.permute(1, 0, 2, 3, 4).flatten(0, 1)
    vutils.save_image(save_image, f'atten_imgs/TwoBiTokenViT/{gt}_{count}_{flag}_{pred}.jpg', nrow=2, normalize=True,
                      scale_each=True)


def blend_mask(image, mask, alpha=0.5, cmap='jet', color='b', color_alpha=1.0):
    # normalize mask
    mask = (mask - mask.min()) / (mask.max() - mask.min() + np.finfo(float).eps)
    if mask.shape != image.shape:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    # get color map
    color_map = plt.get_cmap(cmap)
    mask = color_map(mask)[:, :, :3]
    # convert float to uint8
    mask = (mask * 255).astype(dtype=np.uint8)

    # set the basic color
    basic_color = np.array(colors.to_rgb(color)) * 255
    basic_color = np.tile(basic_color, [image.shape[0], image.shape[1], 1])
    basic_color = basic_color.astype(dtype=np.uint8)
    # blend with basic color
    blended_img = cv2.addWeighted(image, color_alpha, basic_color, 1 - color_alpha, 0)
    # blend with mask
    blended_img = cv2.addWeighted(blended_img, alpha, mask, 1 - alpha, 0)

    return blended_img


def test(opt):
    """ model configuration """
    converter = TokenLabelConverter(opt)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)

    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    # model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)

    # if validators.url(opt.saved_model):
    #     model.load_state_dict(torch.hub.load_state_dict_from_url(opt.saved_model, progress=True, map_location=device))
    # else:
    model_parameters = model.state_dict()
    for name, para in torch.load(opt.saved_model, map_location=device)['net'].items():
        model_parameters[name[7:]] = para
    model.load_state_dict(model_parameters)
    # model.load_state_dict(torch.load(opt.saved_model, map_location=device)['net'])

    opt.exp_name = '_'.join(opt.saved_model.split('/')[1:])
    # print(model)

    """ keep evaluation model and result logs """
    os.makedirs(f'./result/{opt.exp_name}', exist_ok=True)
    os.system(f'cp {opt.saved_model} ./result/{opt.exp_name}/')

    """ setup loss """
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0

    """ evaluation """
    model.eval()
    opt.eval = True
    with torch.no_grad():
        if opt.benchmark_all_eval:  # evaluation with 10 benchmark evaluation datasets
            return benchmark_all_eval(model, criterion, converter, opt)
        else:
            log = open(f'./result/{opt.exp_name}/log_evaluation.txt', 'a')
            AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, opt=opt)
            eval_data, eval_data_log = hierarchical_dataset(root=opt.eval_data, opt=opt)
            evaluation_loader = torch.utils.data.DataLoader(
                eval_data, batch_size=opt.batch_size,
                shuffle=True,
                num_workers=int(opt.workers),
                collate_fn=AlignCollate_evaluation, pin_memory=False)
            _, accuracy_by_best_model, _, _, _, _, _, _ = validation(
                model, criterion, evaluation_loader, converter, opt)
            log.write(eval_data_log)
            print(f'{accuracy_by_best_model:0.3f}')
            log.write(f'{accuracy_by_best_model:0.3f}\n')
            log.close()


# https://github.com/clovaai/deep-text-recognition-benchmark/issues/125
def get_flops(model, opt, converter):
    from thop import profile
    input = torch.randn(1, 1, opt.imgH, opt.imgW).to(device)
    model = model.to(device)
    if opt.Transformer:
        seqlen = converter.batch_max_length
        text_for_pred = torch.LongTensor(1, seqlen).fill_(0).to(device)
        # preds = model(image, text=target, seqlen=converter.batch_max_length)
        MACs, params = profile(model, inputs=(input, text_for_pred, True, seqlen))
    else:
        text_for_pred = torch.LongTensor(1, opt.batch_max_length + 1).fill_(0).to(device)
        # model_ = Model(opt).to(device)
        MACs, params = profile(model, inputs=(input, text_for_pred,))

    flops = 2 * MACs  # approximate FLOPS
    return f'Approximate FLOPS: {flops:0.3f}'


if __name__ == '__main__':
    opt = get_args(is_train=False)

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    if opt.range is not None:
        start_range, end_range = sorted([int(e) for e in opt.range.split('-')])
        print("eval range: ", start_range, end_range)

    if os.path.isdir(opt.model_dir):
        result = []
        model_list = os.listdir(opt.model_dir)
        model_list = [model for model in model_list if model.startswith('iter_')]
        model_list = sorted(model_list, key=lambda x: int(x.split('.')[0].split('_')[-1]), reverse=True)
        err_list = []
        for model in model_list:
            if opt.range is not None:
                num_epoch = int(str(model).split('_')[1].split('.')[0])
                if not (num_epoch >= start_range and num_epoch <= end_range):
                    continue
            opt.saved_model = os.path.join(opt.model_dir, model)
            result.append(test(opt) + [opt.saved_model])
            print('opt.model_path :', opt.saved_model)
        tab_title = ['char_acc', 'bpe_acc', 'wp_acc', 'fused_acc', 'model']
        result = sorted(result, key=lambda x: x[3], reverse=True)
    else:
        opt.saved_model = opt.model_dir
        test(opt)


"""
                fill_value = torch.Tensor([0.7686, 0.8, 0.949])
                mask_pred = torch.cat(
                    [masks[i].squeeze().unsqueeze(2).repeat(1, 1, 3), fill_value.reshape(1, 1, 3).repeat(3, 128, 1),
                     fore_mask[i].squeeze().unsqueeze(2).repeat(1, 1, 3)], dim=0)
                Attn_ = seq_attention_map[i].permute(1, 0)[:len_target[i] - 1]
                Attn_ = torch.nn.functional.interpolate(Attn_.unsqueeze(0).unsqueeze(1).float(),
                                                        size=(len_target[i] - 1, 128),
                                                        scale_factor=None, mode='bilinear',
                                                        align_corners=None).squeeze()
                Attn_ = Attn_ > 0.05
                Attn_ = torch.cat([Attn_, torch.zeros(33 - len_target[i], 128)], dim=0)
                img_seq = torch.cat([image_tensors[i].permute(1, 2, 0), fill_value.reshape(1, 1, 3).repeat(3, 128, 1),
                                     Attn_.unsqueeze(-1).repeat(1, 1, 3)], dim=0)

                grid = torch.nn.functional.interpolate(Single_char_mask[i].unsqueeze(0).float(), size=(
                    32, 128), scale_factor=None, mode='bilinear', align_corners=None).squeeze()

                if (len_target[i] - 1) % 2 == 1:
                    col_num = (len_target[i] - 1) // 2 + 1
                else:
                    col_num = (len_target[i] - 1) // 2

                # grid = grid.unsqueeze(1).float()
                # if grid.dim() == 4 and grid.size(1) == 1:  # single-channel images
                #     grid = torch.cat((grid, grid, grid), 1)
                # grid = grid.permute(0, 2, 3, 1)
                # grid_1 = grid[1]
                # for j in range(2, col_num + 1):
                #     grid_1 = torch.cat([grid_1, grid[j]])

                grid_1 = vutils.make_grid(torch.Tensor(grid[1:col_num + 1]).unsqueeze(1).float(),
                                          normalize=True,
                                          padding=0,
                                          scale_each=True, nrow=col_num)
                grid_1 = grid_1.permute(1, 2, 0)
                grid_2 = vutils.make_grid(torch.Tensor(grid[col_num + 1:len_target[i]]).unsqueeze(1).float(),
                                          normalize=True,
                                          padding=0,
                                          scale_each=True, nrow=col_num)
                grid_2 = grid_2.permute(1, 2, 0)
                fill_zeros = torch.zeros(*grid_1.shape)
                fill_zeros[:, :grid_2.shape[1], :] = grid_2
                grid = torch.cat([grid_1,
                                  fill_value.reshape(1, 1, 3).repeat(3, grid_1.shape[1], 1),
                                  fill_zeros,
                                  ], dim=0)

                char_feature = F.softmax(char_feature_attn[1], dim=1)
                overlaps = []
                ABI_scores = char_feature[i][:len_target[i]].reshape(-1, 16, 64)
                T = ABI_scores.shape[0]
                attn_scores = ABI_scores.detach().cpu().numpy()
                image_numpy = image_tensors[i].detach().cpu().float().numpy()
                if image_numpy.shape[0] == 1:
                    image_numpy = np.tile(image_numpy, (3, 1, 1))
                # x = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2. * 255.0
                x = np.transpose(image_numpy, (1, 2, 0)) * 255.0

                for t in range(T):
                    att_map = attn_scores[t, :, :]  # [feature_H, feature_W, 1] .reshape(32, 8).T
                    # normalize mask
                    att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + np.finfo(float).eps)
                    # x = cv2.resize(x, (64, 16))  # [H, W]
                    att_map = cv2.resize(att_map, (128, 32))
                    att_map = (att_map * 255).astype(np.uint8)
                    heatmap = cv2.applyColorMap(att_map, cv2.COLORMAP_JET)  # [H, W, C]
                    heatmap = heatmap[:, :, ::-1]
                    overlap = cv2.addWeighted(x, 0.5, heatmap, 0.5, 0, dtype=cv2.CV_32F)
                    overlaps.append(overlap)

                char_segmention_1 = vutils.make_grid(torch.Tensor(overlaps[1:col_num + 1]).permute(0, 3, 1, 2),
                                                     normalize=True,
                                                     padding=0,
                                                     scale_each=True, nrow=col_num)
                char_segmention_1 = char_segmention_1.permute(1, 2, 0)
                fill_zeros = torch.zeros(*char_segmention_1.shape)
                char_segmention_2 = vutils.make_grid(torch.Tensor(overlaps[col_num + 1:]).permute(0, 3, 1, 2),
                                                     normalize=True,
                                                     padding=0,
                                                     scale_each=True, nrow=col_num)
                char_segmention_2 = char_segmention_2.permute(1, 2, 0)
                fill_zeros[:, :char_segmention_2.shape[1], :] = char_segmention_2
                char_segmention = torch.cat([char_segmention_1,
                                             fill_value.reshape(1, 1, 3).repeat(3, char_segmention_1.shape[1], 1),
                                             fill_zeros,
                                             ], dim=0)
                # char_segmention = torch.Tensor(np.array(char_segmention)[:, :, ::-1].copy())
                char_attn = torch.cat([grid, fill_value.reshape(1, 1, 3).repeat(67, 3, 1), char_segmention], dim=1)
                final_vis = torch.cat([img_seq, fill_value.reshape(1, 1, 3).repeat(67, 3, 1), mask_pred,
                                       fill_value.reshape(1, 1, 3).repeat(67, 3, 1), char_attn], dim=1)
                final_vis = torch.cat([fill_value.reshape(1, 1, 3).repeat(3, final_vis.shape[1], 1), final_vis,
                                       fill_value.reshape(1, 1, 3).repeat(3, final_vis.shape[1], 1)], dim=0)
                final_vis = torch.cat([fill_value.reshape(1, 1, 3).repeat(final_vis.shape[0], 3, 1), final_vis,
                                       fill_value.reshape(1, 1, 3).repeat(final_vis.shape[0], 3, 1)], dim=1)

                plt.imshow(final_vis.numpy())
                plt.axis('off')
                plt.savefig(f'../demo_imgs/demo4/{num}/{labels[i]}.jpg', bbox_inches='tight', pad_inches=0, dpi=300)
                plt.close()
"""