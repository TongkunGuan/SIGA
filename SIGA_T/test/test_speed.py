import os
import time
import string
import argparse
import re
import PIL
import validators
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def benchmark_all_eval(model, criterion, converter, opt):  # , calculate_infer_time=False):
    """ evaluation with 10 benchmark evaluation datasets """

    if opt.fast_acc:
        # # To easily compute the total accuracy of our paper.
        eval_data_list = ['IC13_857', 'SVT', 'IIIT5k_3000', 'IC15_1811', 'SVTP', 'CUTE80']
    else:
        # The evaluation datasets, dataset order is same with Table 1 in our paper.
        # eval_data_list = ['IIIT5k_3000', 'SVT', 'IC03_860', 'IC03_867', 'IC13_857',
        #                   'IC13_1015', 'IC15_1811', 'IC15_2077', 'SVTP', 'CUTE80']
        eval_data_list = ['ArbitText']

    if opt.calculate_infer_time:
        evaluation_batch_size = 1  # batch_size should be 1 to calculate the GPU inference time per image.
    else:
        evaluation_batch_size = opt.batch_size

    fused_list_accuracy = []
    total_forward_time = 0
    total_evaluation_data_number = 0
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
            collate_fn=AlignCollate_evaluation, pin_memory=True)

        final_accuracy,  infer_time, length_of_data, final_n_correct = validation(
            model, criterion, evaluation_loader, converter, opt)
        fused_list_accuracy.append(f'{final_accuracy:0.3f}')

        total_forward_time += infer_time
        total_evaluation_data_number += len(eval_data)
        # log.write(eval_data_log)

    averaged_forward_time = total_forward_time / total_evaluation_data_number * 1000
    fused_total_accuracy = round(fused_total_correct_number / total_evaluation_data_number * 100, 3)
    params_num = sum([np.prod(p.size()) for p in model.parameters()])

    evaluation_log = 'accuracy: ' + '\n'
    evaluation_log +=  'fused_total_Acc:' + str(fused_total_accuracy) + '\n'
    evaluation_log += f'averaged_infer_time: {averaged_forward_time:0.3f}\t# parameters: {params_num / 1e6:0.3f}'
    if opt.flops:
        evaluation_log += get_flops(model, opt, converter)
    print(evaluation_log)
    log.write(evaluation_log + '\n')
    log.close()

    return fused_total_accuracy


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
    final_n_correct = 0


    length_of_data = 0
    infer_time = 0

    for i, (image_tensors, masks, labels, imgs_path) in tqdm(enumerate(evaluation_loader)):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)
        # For max length prediction
        if opt.Transformer:
            target = converter.encode(labels)
        else:
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
            text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=opt.batch_max_length)

        start_time = time.time()
        Share_pred = model(image, None, target, None, None, is_eval=True)  # final
        forward_time = time.time() - start_time
        target = target[:, 1:]
        cost = criterion(Share_pred.contiguous().view(-1, Share_pred.shape[-1]), target.contiguous().view(-1))
        final_preds_str, final_preds_max_prob = decode(Share_pred, converter, batch_size)

        infer_time += forward_time
        # calculate accuracy & confidence score
        confidence_score_list = []
        for index, gt in enumerate(labels):
            # final
            final_pred = final_preds_str[index]
            final_pred_EOS = final_pred.find('[s]')
            final_pred = final_pred[:final_pred_EOS]
            if final_pred == gt:
                final_n_correct += 1
    final_accuracy = final_n_correct / float(length_of_data) * 100


    return final_accuracy,  infer_time, length_of_data, final_n_correct


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
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)

    # if validators.url(opt.saved_model):
    #     model.load_state_dict(torch.hub.load_state_dict_from_url(opt.saved_model, progress=True, map_location=device))
    # else:
    model.load_state_dict(torch.load(opt.saved_model, map_location=device)['net'])

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
                shuffle=False,
                num_workers=int(opt.workers),
                collate_fn=AlignCollate_evaluation, pin_memory=True)
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
