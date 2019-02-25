import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.autograd import Variable


def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, size_average=size_average, ignore_index=250
    )
    return loss


def multi_scale_cross_entropy2d(input, target, weight=None, size_average=True, scale_weight=None):
    if not isinstance(input, tuple):
        return cross_entropy2d(input=input, target=target, weight=weight, size_average=size_average)

    # Auxiliary training for PSPNet [1.0, 0.4] and ICNet [1.0, 0.4, 0.16]
    if scale_weight is None:  # scale_weight: torch tensor type
        n_inp = len(input)
        scale = 0.4
        scale_weight = torch.pow(scale * torch.ones(n_inp), torch.arange(n_inp).float()).to(
            input[0].device
        )

    loss = 0.0
    for i, inp in enumerate(input):
        loss = loss + scale_weight[i] * cross_entropy2d(
            input=inp, target=target, weight=weight, size_average=size_average
        )

    return loss


def multi_scale_cross_entropy2d_inst(input, target, weight=None, size_average=True, scale_weight=None):
    if not isinstance(input, tuple):
        return cross_entropy2d(input=input, target=target, weight=weight, size_average=size_average)

    # Auxiliary training for PSPNet [1.0, 0.4] and ICNet [1.0, 0.4, 0.16]
    if scale_weight is None:  # scale_weight: torch tensor type
        n_inp = len(input)
        scale = 0.4
        scale_weight = torch.pow(scale * torch.ones(n_inp), torch.arange(n_inp).float()).to(
            input[0].device
        )

    loss = 0.0
    for i, inp in enumerate(input):
        loss = loss + scale_weight[i] * cross_entropy2d(
            input=inp, target=target, weight=weight, size_average=size_average
        )

    return loss


def bootstrapped_cross_entropy2d(input, target, K, weight=None, size_average=True):

    batch_size = input.size()[0]

    def _bootstrap_xentropy_single(input, target, K, weight=None, size_average=True):

        n, c, h, w = input.size()
        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)
        loss = F.cross_entropy(
            input, target, weight=weight, reduce=False, size_average=False, ignore_index=250
        )

        topk_loss, _ = loss.topk(K)
        reduced_topk_loss = topk_loss.sum() / K

        return reduced_topk_loss

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(
            input=torch.unsqueeze(input[i], 0),
            target=torch.unsqueeze(target[i], 0),
            K=K,
            weight=weight,
            size_average=size_average,
        )
    return loss / float(batch_size)


def discrimitive_loss_cs(input, target, delta_var = 0.25, delta_dist = 1.0, delta_reg = 6.0, param_var = 1.0, param_dist = 1.0, param_reg = 0.1):
    # import ipdb
    has_instance_classes = [
            24,
            25,
            26,
            27,
            28,
            31,
            32,
            33,
        ]
    
    bs, feat_dim, h, w = input.size()
    _, ht, wt = target.size()
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    pred_b = input.view(bs, feat_dim, -1)
    gt_b = target.view(bs, -1)
    l_var = Variable(torch.Tensor([0]).to(input.device))
    l_dist = Variable(torch.Tensor([0]).to(input.device))
    l_reg = Variable(torch.Tensor([0]).to(input.device))

    for i in range(bs):
        pred = pred_b[i, ...]
        gt = gt_b[i, ...]
        gt[gt // 1000 < 24] = 0
        gt[gt // 1000 > 33] = 0
        inst_lbls = gt.unique(sorted = True)
        inst_lbls = inst_lbls[1:]
        inst_num = len(inst_lbls)
        inst_count = torch.zeros(inst_lbls.shape).to(input.device)
        inst_means = torch.zeros(feat_dim, inst_num).to(input.device)

        if inst_num == 0:
            continue


        for idx in range(inst_num):
            inst_count[idx] = (gt == inst_lbls[idx]).sum()

        # var term
        for idx in range(inst_num):
            inst_means[:, idx] = torch.sum(pred[:, (gt == inst_lbls[idx])], dim = 1) / inst_count[idx]
            inst_pred_t = pred[:, (gt == inst_lbls[idx])] - inst_means[:, idx][:, None]
            inst_pred_t = torch.norm(inst_pred_t, dim = 0) - delta_var
            inst_pred_t = torch.clamp(inst_pred_t, min = 0) ** 2
            l_var += torch.sum(inst_pred_t) / inst_count[idx]
        l_var /= inst_num


        # dist term
        for lbl in has_instance_classes:
            inst_num_lbl = torch.sum(inst_lbls // 1000 == lbl)
            if inst_num_lbl > 1:
                # feat_dim, inst_num, inst_num
                inst_means_lbl = inst_means[:, inst_lbls // 1000 == lbl]
                means_a = inst_means_lbl.unsqueeze(2).expand(feat_dim, inst_num_lbl, inst_num_lbl)
                means_b = means_a.permute(0, 2, 1)
                diff = means_a - means_b

                margin = Variable(2 * delta_dist * (1.0 - torch.eye(inst_num_lbl))).to(input.device)
                c_dist = torch.sum(torch.clamp(margin - torch.norm(diff,  dim=0), min=0) ** 2)
                l_dist += c_dist / (2 * inst_num_lbl * (inst_num_lbl - 1))
        

  
        # reg term
        l_reg += torch.clamp(torch.mean(torch.norm(inst_means, dim = 0)) - delta_reg, min = 0)

    print("l_var = {:.4f}   l_dist = {:.4f}  l_reg = {:.4f}".format(l_var.item(), l_dist.item(), l_reg.item()))
    l_d =  param_var * l_var + param_dist * l_dist + param_reg * l_reg
    l_d /= bs
    # ipdb.set_trace()
    return l_d
    # return torch.zeros(bs).to(input.device)


def multi_scale_discrimitive_loss_cs(input, target, scale_weight=None):
    if not isinstance(input, tuple):
        return discrimitive_loss_cs(input=input, target=target)

    # Auxiliary training for PSPNet [1.0, 0.4] and ICNet [1.0, 0.4, 0.16]
    if scale_weight is None:  # scale_weight: torch tensor type
        n_inp = len(input)
        scale = 0.4
        scale_weight = torch.pow(scale * torch.ones(n_inp), torch.arange(n_inp).float()).to(
            input[0].device
        )

    loss = 0.0
    for i, inp in enumerate(input):
        loss = loss + scale_weight[i] * discrimitive_loss_cs(
            input=inp, target=target
        )

    return loss