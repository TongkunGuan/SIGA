import os
import sys
import time
import random
import string
import re
import cv2
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
import copy
from torch.utils.tensorboard import SummaryWriter

from utils import Averager, TokenLabelConverter
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from models import Model
from test.test_final import validation
from utils import get_args
import utils_dist as utils

import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
import torchvision.utils as vutils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.detach().cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)
def Heatmap(attention_weight,image,map_size):
    (W, H) = map_size
    overlaps = []
    T = attention_weight.shape[0]
    attention_weight = attention_weight.detach().cpu().numpy()
    x = tensor2im(image)
    for t in range(T):
        att_map = attention_weight[t,:,:] # [feature_H, feature_W, 1]
        att_map = cv2.resize(att_map, (W,H)) # [H, W]
        att_map = (att_map*255).astype(np.uint8)
        heatmap = cv2.applyColorMap(att_map, cv2.COLORMAP_JET) # [H, W, C]
        overlap = cv2.addWeighted(heatmap, 0.6, x, 0.4, 0)
        overlaps.append(overlap)
    return overlaps

def train(opt):
    """ dataset preparation """
    if not opt.data_filtering_off:
        print('Filtering the images containing characters which are not in opt.character')
        print('Filtering the images whose label is longer than opt.batch_max_length')
        # see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L130

    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    opt.eval = False
    train_dataset = Batch_Balanced_Dataset(opt)

    log = open(f'{opt.saved_path}/{opt.exp_name}/log_dataset.txt', 'a')

    val_opt = copy.deepcopy(opt)
    val_opt.eval = True

    os.makedirs(f"./tensorboard", exist_ok=True)
    opt.writer = SummaryWriter(log_dir=f"./tensorboard/{opt.exp_name}")
    
    if opt.sensitive:
        opt.data_filtering_off = True
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, opt=val_opt)
    valid_dataset, _ = hierarchical_dataset(root=opt.valid_data, opt=val_opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size,
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True)
    
    print('-' * 80)
    log.write('-' * 80 + '\n')
    log.close()
        
    """ model configuration """
    converter = TokenLabelConverter(opt)
        
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3

    model = Model(opt)

    print(model)

    # data parallel for multi-GPU
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.gpu], find_unused_parameters=True)
    
    model.train()
    
    if opt.saved_model != '':
        print(f'loading pretrained model from {opt.saved_model}')
        if opt.FT:
            model.load_state_dict(torch.load(opt.saved_model, map_location='cpu')['net'], strict=True)
        else:
            model.load_state_dict(torch.load(opt.saved_model, map_location='cpu')['net'], strict=True)

    """ setup loss """
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
        
    # loss averager
    loss_avg = Averager()

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))

    # setup optimizer
    scheduler = None
    if opt.adam:
        optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)

    if opt.scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=int(opt.num_iter))
        # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1000000)

    """ final options """
    # print(opt)
    with open(f'{opt.saved_path}/{opt.exp_name}/opt.txt', 'a') as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        #print(opt_log)
        opt_file.write(opt_log)
        total_params = int(sum(params_num))
        total_params = f'Trainable network params num : {total_params:,}'
        print(total_params)
        opt_file.write(total_params)

    """ start training """
    start_iter = 0
    if opt.saved_model != '':
        try:
            optimizer.load_state_dict(torch.load(opt.saved_model, map_location='cpu')["optimizer"])
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            scheduler.last_epoch = start_iter
            scheduler._step_count = start_iter
            scheduler.step()
            print(f'continue to train, start_iter: {start_iter}')
        except:
            pass

    start_time = time.time()
    best_accuracy = -1
    iteration = start_iter
            
    print("LR",scheduler.get_last_lr()[0])
    for iteration in tqdm(range(start_iter, opt.num_iter + 1), total=opt.num_iter, position=0, leave=True,):
    # while(True):
        # train part
        image_tensors, mask_tensors, labels = train_dataset.get_batch()
        image = image_tensors.to(device)
        mask = mask_tensors.to(device)
        
        if (opt.Transformer in ["mgp-str"]):
            len_target, char_target = converter.char_encode(labels) 
            bpe_target = converter.bpe_encode(labels)
            wp_target = converter.wp_encode(labels)
            
            char_preds, bpe_preds, wp_preds = model(image, mask)
            
            char_loss = criterion(char_preds.view(-1, char_preds.shape[-1]), char_target.contiguous().view(-1))
            bpe_pred_cost = criterion(bpe_preds.view(-1, bpe_preds.shape[-1]), bpe_target.contiguous().view(-1)) 
            wp_pred_cost = criterion(wp_preds.view(-1, wp_preds.shape[-1]), wp_target.contiguous().view(-1)) 
            cost = char_loss + bpe_pred_cost + wp_pred_cost 

        elif (opt.Transformer in ["char-str"]):
            len_target, char_target = converter.char_encode(labels)

            our_preds, TRBA_preds, glyph_preds, Iter_preds, final_preds, loss, map = model(image, mask, char_target, len_target, iteration)
            char_target = char_target[:, 1:]

            loss_our = criterion(our_preds.view(-1, our_preds.shape[-1]), char_target.contiguous().view(-1))
            loss_TRBA = criterion(TRBA_preds.view(-1, TRBA_preds.shape[-1]), char_target.contiguous().view(-1))
            loss_glyph = criterion(glyph_preds[0].view(-1, glyph_preds[0].shape[-1]), char_target.contiguous().view(-1)) + \
                       criterion(glyph_preds[1].view(-1, glyph_preds[1].shape[-1]), char_target.contiguous().view(-1))
            loss_Iter = criterion(Iter_preds[0].view(-1, Iter_preds[0].shape[-1]), char_target.contiguous().view(-1)) + \
                     criterion(Iter_preds[1].view(-1, Iter_preds[1].shape[-1]), char_target.contiguous().view(-1))
            loss_final = criterion(final_preds.view(-1, final_preds.shape[-1]), char_target.contiguous().view(-1))
            if loss is not None:
                predmask_loss = loss[0].mean()
                Correct_loss = loss[1].mean()
                loss_low, loss_middle = loss[2]
                Segmention_loss = loss_low.mean() + loss_middle.mean()
            cost = loss_our + loss_TRBA + loss_glyph + predmask_loss + 0.15 * Correct_loss + Segmention_loss + loss_Iter + loss_final

            if utils.is_main_process() and iteration % opt.showInterval == 0:
                opt.writer.add_scalar('loss_our', loss_our, iteration)
                opt.writer.add_scalar('loss_TRBA', loss_TRBA, iteration)
                opt.writer.add_scalar('loss_glyph', loss_glyph, iteration)
                opt.writer.add_scalar('loss_Iter', loss_Iter, iteration)
                opt.writer.add_scalar('loss_final', loss_final, iteration)
                opt.writer.add_scalar('Pred_mask_loss', predmask_loss, iteration)
                opt.writer.add_scalar('Correct Location', Correct_loss, iteration)
                opt.writer.add_scalar('Segmention_loss', Segmention_loss, iteration)
                opt.writer.add_scalar('lr', scheduler.get_last_lr()[0], iteration)

                seq_attention_map, char_feature_attn, Single_char_mask, fore_mask = map
                input_image = image[0]
                opt.writer.add_image('Input image', input_image, iteration)

                """ GT mask """
                gt_mask = mask[0]
                opt.writer.add_image('GT mask', gt_mask, iteration)

                """ Pred mask """
                Pred_mask = fore_mask[0]
                Attn = seq_attention_map[0].permute(1, 0).unsqueeze(0).unsqueeze(0)
                Loc_inter = nn.functional.interpolate(Attn.float(), size=(26, opt.imgW),
                                                      scale_factor=None, mode='bilinear', align_corners=None).squeeze(0)
                Mask_Loc = torch.cat([Pred_mask, Loc_inter], dim=1)
                opt.writer.add_image('Mask_Loc Maps', Mask_Loc, iteration)

                """ Char Segmention """
                Softmax_attn = F.softmax(char_feature_attn[0], dim=1)[:, 1:, :, :]
                char_segmention = Softmax_attn[0]
                char_segmention = Heatmap(char_segmention, input_image, (opt.imgW, opt.imgH))
                char_segmention = vutils.make_grid(torch.Tensor(char_segmention).permute(0, 3, 1, 2), normalize=True,
                                                   scale_each=True, nrow=5)
                opt.writer.add_image('char_segmention_Middle Maps', char_segmention, iteration)

                Softmax_attn = F.softmax(char_feature_attn[1], dim=1)[:, 1:, :, :]
                char_segmention = Softmax_attn[0]
                char_segmention = Heatmap(char_segmention, input_image, (opt.imgW, opt.imgH))
                char_segmention = vutils.make_grid(torch.Tensor(char_segmention).permute(0, 3, 1, 2), normalize=True,
                                                   scale_each=True, nrow=5)
                opt.writer.add_image('char_segmention_Low Maps', char_segmention, iteration)


                Single_char_mask = Single_char_mask[0].unsqueeze(1)
                Single_char_mask = vutils.make_grid(Single_char_mask.float(), normalize=True)
                opt.writer.add_image('Single_char_mask Maps', Single_char_mask, iteration)

        model.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
        optimizer.step()

        loss_avg.add(cost)
        # torch.cuda.synchronize()
        # validation part
        if utils.is_main_process() and (iteration % opt.valInterval == 0 or iteration == 0): # To see training progress, we also conduct validation when 'iteration == 0'
            elapsed_time = time.time() - start_time
            # for log
            print("LR",scheduler.get_last_lr()[0])
            with open(f'{opt.saved_path}/{opt.exp_name}/log_train.txt', 'a') as log:
                model.eval()
                print('start_evaluation!')
                with torch.no_grad():
                    valid_loss, current_accuracys, char_preds, confidence_score, labels, infer_time, length_of_data, _ = validation(
                        model, criterion, valid_loader, converter, opt)
                    char_accuracy = current_accuracys[0]
                    gly_accuracy = current_accuracys[1]
                    Iter_accuracy = current_accuracys[2]
                    final_accuracy = current_accuracys[3]
                    out_accuracy = current_accuracys[4]
                    cur_best = max(char_accuracy, gly_accuracy, Iter_accuracy, final_accuracy, out_accuracy)
                model.train()

                loss_log = f'[{iteration+1}/{opt.num_iter}] LR: {scheduler.get_last_lr()[0]:0.5f}, Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
                loss_avg.reset()
                current_model_log = f'{"char_accuracy":17s}: {char_accuracy:0.3f}, ' \
                                    f'{"bpe_accuracy":17s}: {gly_accuracy:0.3f}, ' \
                                    f'{"wp_accuracy":17s}: {Iter_accuracy:0.3f}, ' \
                                    f'{"fused_accuracy":17s}: {final_accuracy:0.3f}, ' \
                                    f'{"out_accuracy":17s}: {out_accuracy:0.3f} '

                opt.writer.add_scalar('accuracy', cur_best, iteration)

                # keep best accuracy model (on valid dataset)
                if cur_best > best_accuracy:
                    best_accuracy = cur_best
                    checkpoint = {
                        'net': model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        'iteration': iteration,
                    }
                    torch.save(checkpoint, f'{opt.saved_path}/{opt.exp_name}/best_accuracy.pth')
                best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}'

                loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                print(loss_model_log)
                log.write(loss_model_log + '\n')

                # show some predicted results
                dashed_line = '-' * 80
                head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                for gt, pred, confidence in zip(labels[:5], char_preds[:5], confidence_score[:5]):
                    if opt.Transformer:
                        pred = pred[:pred.find('[s]')]
                    elif 'Attn' in opt.Prediction:
                        gt = gt[:gt.find('[s]')]
                        pred = pred[:pred.find('[s]')]

                    # To evaluate 'case sensitive model' with alphanumeric and case insensitve setting.
                    if opt.sensitive and opt.data_filtering_off:
                        pred = pred.lower()
                        gt = gt.lower()
                        alphanumeric_case_insensitve = '0123456789abcdefghijklmnopqrstuvwxyz'
                        out_of_alphanumeric_case_insensitve = f'[^{alphanumeric_case_insensitve}]'
                        pred = re.sub(out_of_alphanumeric_case_insensitve, '', pred)
                        gt = re.sub(out_of_alphanumeric_case_insensitve, '', gt)

                    predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
                predicted_result_log += f'{dashed_line}'
                print(predicted_result_log)
                log.write(predicted_result_log + '\n')

        # save model per 1e+5 iter.
        if utils.is_main_process() and (iteration + 1) % 1e+5 == 0:
            checkpoint = {
                'net': model.state_dict(),
                "optimizer": optimizer.state_dict(),
                'iteration': iteration,
            }
            torch.save(
                checkpoint, f'{opt.saved_path}/{opt.exp_name}/iter_{iteration+1}.pth')

        if (iteration + 1) == opt.num_iter:
            print('end the training')
            sys.exit()
            # iteration += 1
        if scheduler is not None:
            scheduler.step()

if __name__ == '__main__':

    opt = get_args()

    if not opt.exp_name:
        opt.exp_name = f'{opt.TransformerModel}' if opt.Transformer else f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'

    opt.exp_name += f'-Seed{opt.manualSeed}'

    os.makedirs(f'{opt.saved_path}/{opt.exp_name}', exist_ok=True)

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).
    
    utils.init_distributed_mode(opt)

    print(opt)
    
    """ Seed and GPU setting """
    
    seed = opt.manualSeed + utils.get_rank()
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    
    train(opt)

"""
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=3 --nnodes=1 --master_port 29501 train_final_dist.py --train_data /home/jcc/GTK/TextRecognitionDataset/training/label/Synth/ --valid_data /home/jcc/GTK/TextRecognitionDataset/evaluation/benchmark/  --select_data MJ-ST  --batch_ratio 0.5-0.5  --Transformer char-str --TransformerModel=char_str_base_patch4_3_32_128 --imgH 32 --imgW 128 --manualSeed=226 --workers=12 --isrand_aug --scheduler --batch_size=48 --rgb --saved_path ./ --exp_name char_str_patch4_3_32_128 --valInterval 5000 --num_iter 1400000 --lr 1
"""