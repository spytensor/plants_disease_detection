import shutil
import sys
import os
import json
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from config import config


def save_checkpoint(state, is_best, fold):
    filename = config.weights + config.model_name + os.sep + str(fold) + os.sep + "_checkpoint.pth.tar"
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, config.best_models + config.model_name + os.sep + str(fold) + os.sep + 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  # stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None:
            mode = 'w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1):
        if '\r' in message:
            is_file = 0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        pass


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    lr = lr[0]
    return lr


def time_to_str(t, mode='min'):
    if mode == 'min':
        t = int(t) / 60
        hr = t // 60
        min = t % 60
        return '%2d hr %02d min' % (hr, min)

    elif mode == 'sec':
        t = int(t)
        min = t // 60
        sec = t % 60
        return '%2d min %02d sec' % (min, sec)

    else:
        raise NotImplementedError


class FocalLoss(nn.Module):
    """Focal Loss，逐样本计算难易加权。

    参考 Lin et al., "Focal Loss for Dense Object Detection".
    关键点：cross_entropy 必须用 reduction='none' 拿到每个样本的 loss，
    再乘以 (1 - pt) ** gamma 的调制因子，否则退化成普通交叉熵。
    """

    def __init__(self, gamma=2, alpha=0.25, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, output, target):
        # 逐样本的 -log(pt)
        logpt = -F.cross_entropy(output, target, reduction="none")
        pt = torch.exp(logpt)
        focal_loss = -self.alpha * (1 - pt) ** self.gamma * logpt

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)
