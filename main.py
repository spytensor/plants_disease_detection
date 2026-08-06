import os
import random
import time
import json
import torch
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from torch import nn, optim
from config import config
from collections import OrderedDict
from torch.utils.data import DataLoader
from dataset.dataloader import *
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
from models.model import *
from utils import *

# 1. set random seed and device
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)

device = torch.device(config.device)
if device.type == "cuda":
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = True

use_amp = config.use_amp and device.type == "cuda"
warnings.filterwarnings('ignore')


# 2. evaluate func
def evaluate(val_loader, model, criterion):
    # 2.1 define meters
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    # 2.2 switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.to(device)
            target = torch.as_tensor(target, dtype=torch.long, device=device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                output = model(input)
                loss = criterion(output, target)

            # 2.2.1 measure accuracy and record loss
            precision1, precision2 = accuracy(output, target, topk=(1, 2))
            losses.update(loss.item(), input.size(0))
            top1.update(precision1[0].item(), input.size(0))
            top2.update(precision2[0].item(), input.size(0))

    return [losses.avg, top1.avg, top2.avg]


# 3. test model on public dataset and save the probability matrix
def test(test_loader, model, folds):
    csv_map = OrderedDict({"filename": [], "probability": []})
    model.eval()
    with open("./submit/baseline.json", "w", encoding="utf-8") as f:
        submit_results = []
        for i, (input, filepath) in enumerate(tqdm(test_loader)):
            # 只保留文件名
            filepath = [os.path.basename(x) for x in filepath]
            with torch.no_grad():
                image_var = input.to(device)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    y_pred = model(image_var)
                    smax_out = nn.functional.softmax(y_pred, dim=1)
            # 保存每张图的概率向量
            csv_map["filename"].extend(filepath)
            for output in smax_out:
                prob = ";".join([str(i) for i in output.data.tolist()])
                csv_map["probability"].append(prob)
        result = pd.DataFrame(csv_map)
        result["probability"] = result["probability"].map(lambda x: [float(i) for i in x.split(";")])
        for index, row in result.iterrows():
            pred_label = np.argmax(row['probability'])
            # move.py 中删除了原始的 44、45 两类并把 >45 的类别 -2，
            # 这里做逆映射把预测类别还原回官方原始编号：>=44 的 +2
            if pred_label > 43:
                pred_label = pred_label + 2
            submit_results.append({"image_id": row['filename'], "disease_class": pred_label})
        json.dump(submit_results, f, ensure_ascii=False, cls=MyEncoder)


# 4. main function
def main():
    fold = 0
    # 4.1 mkdirs
    for d in [config.submit, config.weights, config.best_models, config.logs]:
        os.makedirs(d, exist_ok=True)
    os.makedirs(config.weights + config.model_name + os.sep + str(fold) + os.sep, exist_ok=True)
    os.makedirs(config.best_models + config.model_name + os.sep + str(fold) + os.sep, exist_ok=True)

    # 4.2 get model, optimizer, criterion
    model = get_net()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, amsgrad=True, weight_decay=config.weight_decay)
    if config.use_focal_loss:
        criterion = FocalLoss().to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    log = Logger()
    log.open(config.logs + "log_train.txt", mode="a")
    log.write("\n----------------------------------------------- [START %s] %s\n\n" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))

    # 4.3 params for restart
    start_epoch = 0
    best_precision1 = 0
    best_precision_save = 0
    resume = False

    # 4.4 restart the training process
    if resume:
        checkpoint = torch.load(config.best_models + str(fold) + "/model_best.pth.tar", map_location=device, weights_only=False)
        start_epoch = checkpoint["epoch"]
        fold = checkpoint["fold"]
        best_precision1 = checkpoint["best_precision1"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    # 4.5 get files and split
    train_ = get_files(config.train_data, "train")
    test_files = get_files(config.test_data, "test")

    train_data_list, val_data_list = train_test_split(train_, test_size=0.15, stratify=train_["label"], random_state=config.seed)
    # 4.5.4 load dataset
    train_dataloader = DataLoader(ChaojieDataset(train_data_list), batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True, num_workers=config.num_workers)
    val_dataloader = DataLoader(ChaojieDataset(val_data_list, train=False), batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=False, num_workers=config.num_workers)
    test_dataloader = DataLoader(ChaojieDataset(test_files, test=True), batch_size=1, shuffle=False, pin_memory=False, num_workers=config.num_workers)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 4.5.5.1 define metrics
    train_losses = AverageMeter()
    train_top1 = AverageMeter()
    train_top2 = AverageMeter()
    valid_loss = [np.inf, 0, 0]
    model.train()
    # logs
    log.write('** start training here! **\n')
    log.write('                           |------------ VALID -------------|----------- TRAIN -------------|------Accuracy------|------------|\n')
    log.write('lr       iter     epoch    | loss   top-1  top-2            | loss   top-1  top-2           |    Current Best    | time       |\n')
    log.write('-------------------------------------------------------------------------------------------------------------------------------\n')
    # 4.5.5 train
    start = timer()
    for epoch in range(start_epoch, config.epochs):
        # train
        for iter, (input, target) in enumerate(train_dataloader):
            model.train()
            input = input.to(device)
            target = torch.as_tensor(target, dtype=torch.long, device=device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                output = model(input)
                loss = criterion(output, target)

            precision1_train, precision2_train = accuracy(output, target, topk=(1, 2))
            train_losses.update(loss.item(), input.size(0))
            train_top1.update(precision1_train[0].item(), input.size(0))
            train_top2.update(precision2_train[0].item(), input.size(0))
            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr = get_learning_rate(optimizer)
            print('\r', end='', flush=True)
            print('%0.4f %5.1f %6.1f        | %0.3f  %0.3f  %0.3f         | %0.3f  %0.3f  %0.3f         |         %s         | %s' % (
                lr, iter / len(train_dataloader) + epoch, epoch,
                valid_loss[0], valid_loss[1], valid_loss[2],
                train_losses.avg, train_top1.avg, train_top2.avg, str(best_precision_save),
                time_to_str((timer() - start), 'min'))
                , end='', flush=True)
        # StepLR 按 epoch 更新，放在 epoch 末尾、不再传 epoch 参数（旧用法已废弃）
        scheduler.step()
        # evaluate
        lr = get_learning_rate(optimizer)
        valid_loss = evaluate(val_dataloader, model, criterion)
        is_best = valid_loss[1] > best_precision1
        best_precision1 = max(valid_loss[1], best_precision1)
        best_precision_save = best_precision1
        save_checkpoint({
            "epoch": epoch + 1,
            "model_name": config.model_name,
            "state_dict": model.state_dict(),
            "best_precision1": best_precision1,
            "optimizer": optimizer.state_dict(),
            "fold": fold,
            "valid_loss": valid_loss,
        }, is_best, fold)
        print("\r", end="", flush=True)
        log.write('%0.4f %5.1f %6.1f        | %0.3f  %0.3f  %0.3f          | %0.3f  %0.3f  %0.3f         |         %s         | %s' % (
            lr, 0 + epoch, epoch,
            valid_loss[0], valid_loss[1], valid_loss[2],
            train_losses.avg, train_top1.avg, train_top2.avg, str(best_precision_save),
            time_to_str((timer() - start), 'min'))
        )
        log.write('\n')
        time.sleep(0.01)
    best_model = torch.load(config.best_models + os.sep + config.model_name + os.sep + str(fold) + os.sep + 'model_best.pth.tar', map_location=device, weights_only=False)
    model.load_state_dict(best_model["state_dict"])
    test(test_dataloader, model, fold)


if __name__ == "__main__":
    main()
