import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.utils import DiceLoss
from torchvision import transforms
import torchvision
import torch.nn.functional as F
from datasets.dataset_synapse import Synapse_dataset, RandomGenerator, RandomGenerator_DINO

class KDloss(nn.Module):
    def __init__(self, lambda_x):
        super(KDloss, self).__init__()
        self.lambda_x = lambda_x

    def inter_fd(self, f_s, f_t):
        s_C, t_C, s_H, t_H = f_s.shape[1], f_t.shape[1], f_s.shape[2], f_t.shape[2]
        if s_H > t_H:
            f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
        elif s_H < t_H:
            f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))

        idx_s = random.sample(range(s_C), min(s_C, t_C))
        idx_t = random.sample(range(t_C), min(s_C, t_C))
        inter_fd_loss = F.mse_loss(f_s[:, idx_s, :, :], f_t[:, idx_t, :, :].detach())
        return inter_fd_loss

    def intra_fd(self, f_s):
        sorted_s, indices_s = torch.sort(F.normalize(f_s, p=2, dim=(2, 3)).mean([0, 2, 3]), dim=0, descending=True)
        f_s = torch.index_select(f_s, 1, indices_s)
        intra_fd_loss = F.mse_loss(f_s[:, 0:f_s.shape[1]//2, :, :], f_s[:, f_s.shape[1]//2:, :, :])
        return intra_fd_loss

    def forward(self, feature, feature_decoder, final_up):
        f1_0, f2_0, f3_0, f4_0 = feature
        f1_d_0, f2_d_0, f3_d_0 = feature_decoder
        final_layer = final_up

        loss = (self.intra_fd(f1_0) + self.intra_fd(f2_0) + self.intra_fd(f3_0) + self.intra_fd(f4_0)) / 4
        loss += (self.intra_fd(f1_d_0) + self.intra_fd(f2_d_0) + self.intra_fd(f3_d_0)) / 3
        loss += (self.inter_fd(f1_d_0, final_layer) + self.inter_fd(f2_d_0, final_layer) + self.inter_fd(f3_d_0, final_layer) +
                 self.inter_fd(f1_0, final_layer) + self.inter_fd(f2_0, final_layer) + self.inter_fd(f3_0, final_layer) + self.inter_fd(f4_0, final_layer)) / 7
        return loss * self.lambda_x

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    schedule = np.concatenate((warmup_schedule, schedule))
    return schedule

def validate(model, val_loader, num_classes):
    from utils.utils import calculate_metric_percase_dice
    model.eval()
    dice_scores = []

    with torch.no_grad():
        for batch in val_loader:
            image, label = batch['image'].cuda(), batch['label'].cuda()
            output = model(image)
            pred = torch.argmax(torch.softmax(output, dim=1), dim=1)

            for i in range(1, num_classes):
                pred_i = (pred == i).cpu().numpy().astype(np.uint8)
                label_i = (label == i).cpu().numpy().astype(np.uint8)
                dice, _, _, _ = calculate_metric_percase_dice(pred_i, label_i)
                dice_scores.append(dice)

    model.train()
    return np.mean(dice_scores)


def trainer_synapse(args, model, snapshot_path, val_list_filename='val.txt'):
    logging.basicConfig(filename=os.path.join(snapshot_path, "log.txt"), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose([
                                   RandomGenerator(output_size=[args.img_size, args.img_size])]),
                               transform_dino=transforms.Compose([
                                   RandomGenerator_DINO(output_size=[args.img_size, args.img_size])]))

    db_val = Synapse_dataset(base_dir=args.root_path.replace("train_npz", "val_npz"), list_dir=args.list_dir, split="val",
                             transform=transforms.Compose([
                                 RandomGenerator(output_size=[args.img_size, args.img_size])]))

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.001)
    writer = SummaryWriter(snapshot_path + '/log')
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    best_dice = 0.0
    iter_num = 0
    momentum_schedule = cosine_scheduler(0.996, 1, max_iterations, len(trainloader))

    for epoch_num in tqdm(range(max_epoch), ncols=70):
        for sampled_batch in trainloader:
            image_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch.long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/dice_loss', loss_dice, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

        val_dice = validate(model, valloader, num_classes)
        log_str = f"🧪 Epoch {epoch_num} Validation Dice: {val_dice:.4f}"
        tqdm.write(log_str)

        latest_path = os.path.join(snapshot_path, 'latest_checkpoint.pth')
        torch.save({
            'epoch': epoch_num,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, latest_path)

        if val_dice > best_dice:
            best_dice = val_dice
            best_path = os.path.join(snapshot_path, 'best_model.pth')
            torch.save(model.state_dict(), best_path)

    logging.info(f"🏆 Saved best model: epoch {epoch_num}, dice {val_dice:.4f}, to {best_path}")
    writer.close()
    return "Training Finished!"
