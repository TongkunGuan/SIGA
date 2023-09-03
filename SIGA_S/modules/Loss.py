import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.alpha = torch.Tensor([alpha, 1 - alpha])
        self.size_average = size_average

    def forward(self, input, target):
        """
        input: #(len, 2 , h*w)
        target: #(len, 1, h*w)
        """
        HW = input.size()[-1]
        if input.dim() > 2:
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)
        # logpt = F.log_softmax(input)
        logpt = torch.log(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
                at = self.alpha.gather(0, target.data.view(-1))
                logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        return loss.sum() / HW


class SegLoss(nn.Module):
    def __init__(self, loss_seg=False):
        super(SegLoss, self).__init__()
        self.num_classes = 1
        self.loss_seg = loss_seg
        self.gts_loss = torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()
        self.m = nn.Sigmoid()

    def cross_entropy(self, global_text_segs, gts_masks, bool_):
        if global_text_segs.shape[2] == gts_masks.shape[2]:
            global_mask = gts_masks
        else:
            global_mask = nn.functional.interpolate(gts_masks.float(),
                                                    size=(global_text_segs.shape[2], global_text_segs.shape[3]),
                                                    scale_factor=None, mode='bilinear', align_corners=None)
        input_global_masks = global_mask.view([-1]).long()
        pred_masks = global_text_segs.permute(0, 2, 3, 1).contiguous().view([-1, 2])
        loss = F.cross_entropy(pred_masks, input_global_masks, reduce=bool_)
        return loss

    def forward(self, seg_mask, gts_masks, bool_):
        if isinstance(seg_mask, list):
            cel_loss = self.cross_entropy(seg_mask[0], gts_masks) + self.cross_entropy(seg_mask[1], gts_masks)
        else:
            cel_loss = self.cross_entropy(seg_mask, gts_masks, bool_)
        return cel_loss


class Correct_1D_Location(nn.Module):
    def __init__(self):
        super(Correct_1D_Location, self).__init__()
        self.crossentropy = torch.nn.CrossEntropyLoss()

    def DifferentiableBinarization(self, Attn0):
        pha = 1 / (1 + torch.exp_(-1 * 85 * (Attn0 - 0.08)))
        # (70, 0.1, 32) (85, 0.08, 40)
        # attn = torch.cat([Attn0, pha], dim=0)
        # plt.imshow(attn.cpu().detach().numpy())
        return pha

    def Learn_Irrelevant_vector(self, Attn0):
        loss = 0.0
        num = Attn0.shape[0]
        for i in range(num - 1):
            loss += torch.pow((Attn0[i] * Attn0[i + 1]), 2).sum()
            if i + 2 < num:
                loss += torch.pow((Attn0[i] * Attn0[i + 2:]), 2).sum()
        return loss / num

    def make_seg(self, Attn0, gts_masks, length):
        Attn0 = F.sigmoid(Attn0)
        seq1_attention_Interpolation = nn.functional.interpolate(Attn0.unsqueeze(1).float(), size=(
            Attn0.shape[1], gts_masks.shape[3]), scale_factor=None, mode='bilinear', align_corners=None).squeeze()
        All_Single_char = []
        for i, index in enumerate(length - 1):
            value = seq1_attention_Interpolation[i, :index]
            # value = self.Normalization(value)
            Single_char = value.unsqueeze(1) * gts_masks[i]  # (b,k,h,w)
            Single_char = Single_char.sum(0)
            Single_char = torch.clamp(Single_char, 0., 1.)
            All_Single_char.append(
                torch.cat([(1 - Single_char).unsqueeze(0), Single_char.unsqueeze(0)], 0).unsqueeze(0))
        return torch.cat(All_Single_char, 0), None

    def make_seg_irrelevant(self, Attn0, gts_masks, length):
        seq1_attention_Interpolation = nn.functional.interpolate(Attn0.unsqueeze(1).float(), size=(
            Attn0.shape[1], gts_masks.shape[3]), scale_factor=None, mode='bilinear', align_corners=None).squeeze()
        All_Single_char = []
        Irrelevant_loss = 0.0
        for i, index in enumerate(length - 1):
            value = seq1_attention_Interpolation[i, :index]
            Irrelevant_loss += self.Learn_Irrelevant_vector(value)
            value = self.DifferentiableBinarization(value)
            # value = self.Normalization(value)
            Single_char = value.unsqueeze(1) * gts_masks[i]  # (b,k,h,w)
            if Single_char[0].sum() == 0:
                print('yes')
            Single_char = Single_char.sum(0)
            Single_char = torch.clamp(Single_char, 0., 1.)
            All_Single_char.append(
                torch.cat([(1 - Single_char).unsqueeze(0), Single_char.unsqueeze(0)], 0).unsqueeze(0))
        return torch.cat(All_Single_char, 0), Irrelevant_loss

    def forward(self, Attn0, backfore_cls, length, iteration):
        seg_mask = backfore_cls.clone().detach()
        fore_mask = (seg_mask[:, 1, :, :] > 0.5).int().unsqueeze(1)
        Single_char, Irrelevant_loss = self.make_seg_irrelevant(Attn0, fore_mask, length)
        # loss = torch.mean(torch.abs(Single_char - fore_mask.squeeze()).float())
        b, c, h, w = Single_char.shape
        loss = self.crossentropy(Single_char.permute(0, 2, 3, 1).reshape(b * h * w, c), fore_mask.view(-1).long())
        return loss + Irrelevant_loss



class Char_Segmention_Loss_Attn_Attn_KL(nn.Module):
    def __init__(self):
        super(Char_Segmention_Loss_Attn_Attn_KL, self).__init__()
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((1, None))  # Transform final (imgH/16-1) -> 1
        weight = torch.Tensor([1.000,
                               1.000, 1.013, 1.058, 1.323, 1.564, 1.831, 2.210, 2.859, 2.859, 2.859,
                               2.859, 2.859, 2.859, 2.210, 1.831, 1.564, 1.323, 1.058, 1.013, 1.000,
                               1.000, 1.000, 1.000, 1.000, 1.000, 1.000])
        self.crossentropy = torch.nn.CrossEntropyLoss(weight = weight)

    def make_char_mask(self, seq1_attention_map, fore_mask, alpha):
        seq1_attention_Interpolation = self.Process_Attn(seq1_attention_map, fore_mask.shape[-1], alpha)

        Single_char = seq1_attention_Interpolation.unsqueeze(2) * fore_mask  # (b,k,h,w)
        Single_char = torch.cat([1 - fore_mask, Single_char], dim=1)
        return Single_char

    def KL_div(self, char_feature_attn):
        b, c, h, w = char_feature_attn[1].shape
        Char_High = nn.functional.interpolate(char_feature_attn[0], size=(
            h, w), scale_factor=None, mode='bilinear', align_corners=None).squeeze()
        Char_Low = nn.functional.interpolate(char_feature_attn[2], size=(
            h, w), scale_factor=None, mode='bilinear', align_corners=None).squeeze()
        Char_Middle = F.log_softmax(char_feature_attn[1], dim=1).view(-1, c)
        loss = F.kl_div(Char_Middle, F.softmax(Char_High, dim=1).view(-1, c), reduction='mean') + \
               F.kl_div(Char_Middle, F.softmax(Char_Low, dim=1).view(-1, c), reduction='mean')
        return loss * 2

    def Process_Attn(self, Attn_, size, alpha):
        Loc_inter = nn.functional.interpolate(Attn_.unsqueeze(1).float(), size=(26, size),
                                              scale_factor=None, mode='bilinear',
                                              align_corners=None).squeeze()
        return (Loc_inter > alpha).int()

    def clear_noise(self, seq1_attention_map, length):
        for number, index in enumerate(length - 1):
            seq1_attention_map[number, index:, :] = torch.Tensor([0.]).to(seq1_attention_map.device)
        return seq1_attention_map

    def Dice_multi_classification(self, prob, target, c):
        dice_loss = 0.0
        smooth = 1
        num = target.size(0)
        for i, len in enumerate(c - 1):
            if len == 1:
                m1 = prob[i, 1, :, :].view(len, -1)
                m2 = target[i, 1, :, :].view(len, -1)
            else:
                m1 = prob[i, 1:len + 1, :, :].view(len, -1)
                m2 = target[i, 1:len + 1, :, :].view(len, -1)
            intersection = (m1 * m2)
            score = ((2. * intersection.sum(1)) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
            loss = 1 - score
            dice_loss += loss.mean()
        return dice_loss / num

    def multiclass_focalloss(self, prob, target, c):
        loss = 0.0
        num = target.size(0)
        m = FocalLoss()
        for i, len in enumerate(c - 1):
            if len == 1:
                m1 = prob[i, 1, :, :].view(-1, 1)
                m1 = torch.cat([1 - m1, m1], dim=1)
                m2 = target[i, 1, :, :]
                loss += m(m1, m2)
            else:
                m1 = prob[i, 1:len, :, :].view(len - 1, 1, -1)
                m1 = torch.cat([1 - m1, m1], dim=1)
                m2 = target[i, 1:len, :, :]
                loss += m(m1, m2)
        return loss / num

    def loss(self, char_feature, Single_char_mask, fore_mask, length):
        c = char_feature.shape[1]
        CEL_27 = self.crossentropy(char_feature.permute(0, 2, 3, 1).reshape(-1, c),
                                   torch.argmax(Single_char_mask, dim=1).view(-1))

        Soft_char = F.softmax(char_feature, dim=1)
        Dice_27 = self.Dice_multi_classification(Soft_char, Single_char_mask, length)

        loss = CEL_27 + Dice_27
        return loss

    def forward(self, char_feature_attn, seq1_attention_map, fore_mask, length, alpha):
        b, c, h_middle, w_middle = char_feature_attn[0].shape
        _, _, h_low, w_low = char_feature_attn[1].shape

        seq1_attention_map = seq1_attention_map.clone().detach()
        fore_mask = fore_mask.clone().detach()
        seq1_attention_map = self.clear_noise(seq1_attention_map, length)

        fore_mask_low = nn.functional.interpolate(fore_mask.float(), size=(
            h_low, w_low), scale_factor=None, mode='bilinear', align_corners=None)
        fore_mask_low = (fore_mask_low >= 0.4).long()
        Single_char_mask_low = self.make_char_mask(seq1_attention_map, fore_mask_low, alpha)
        loss_low = self.loss(char_feature_attn[1], Single_char_mask_low, fore_mask_low, length)

        fore_mask_middle = nn.functional.interpolate(fore_mask.float(), size=(
            h_middle, w_middle), scale_factor=None, mode='bilinear', align_corners=None)
        fore_mask_middle = (fore_mask_middle >= 0.4).long()
        Single_char_mask_middle = self.make_char_mask(seq1_attention_map, fore_mask_middle, alpha)
        loss_middle = self.loss(char_feature_attn[0], Single_char_mask_middle, fore_mask_middle, length)
        return (loss_low, loss_middle), Single_char_mask_low


class STR_Loss(nn.Module):
    def __init__(self):
        super(STR_Loss, self).__init__()
        self.seg_loss = SegLoss()
        self.iou_loss = Char_Segmention_Loss_Attn_Attn_KL()
        self.correct_loc = Correct_1D_Location()

    def forward(self, masks, backfore_feature, seq1_attention_map, char_feature_attn, length, iteration, alpha):
        """ Two categories mask loss """
        nb, nc, nh, nw = masks.shape
        masks = (masks > 0.5).int()
        backfore_feature_softmax = F.softmax(backfore_feature, dim=1)
        if iteration > 20000:
            predmask_loss = self.seg_loss(backfore_feature_softmax, masks, False)
            try:
                value = predmask_loss.view(nb, nc * nh * nw, 1).mean(1)
                index = torch.where(value < 1.0)
                predmask_loss = value[index].mean()
            except:
                predmask_loss = predmask_loss.mean()
                print('error')
        else:
            predmask_loss = self.seg_loss(backfore_feature_softmax, masks, True)
        fore_mask = (backfore_feature_softmax[:, 1, :, :] > 0.5).int().unsqueeze(1)

        """ Correct 1D Location"""
        Attn = seq1_attention_map.permute(0, 2, 1)
        Correct_loss = self.correct_loc(Attn, backfore_feature_softmax, length, iteration)

        """ Multi categories mask loss """
        Mutual_exclusion_loss, Single_char_mask = self.iou_loss(char_feature_attn, Attn,
                                                                fore_mask, length, alpha)

        Softmax_classification_Middle = F.softmax(char_feature_attn[0], dim=1)[:, 1:, :, :]
        Softmax_classification_Low = F.softmax(char_feature_attn[1], dim=1)[:, 1:, :, :]

        return (predmask_loss, Correct_loss, Mutual_exclusion_loss), \
               Single_char_mask, Softmax_classification_Middle, Softmax_classification_Low

class MultiLosses(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss().to(device)

    def _flatten(self, sources, lengths):
        return torch.cat([t[:l] for t, l in zip(sources, lengths)])

    def forward(self, pt_logits, gt_labels, gt_lengths):
        assert pt_logits.shape[0] % gt_labels.shape[0] == 0
        flat_gt_labels = self._flatten(gt_labels, gt_lengths)
        flat_pt_logits = self._flatten(pt_logits, gt_lengths)

        loss = self.ce(flat_pt_logits, flat_gt_labels)
        return loss