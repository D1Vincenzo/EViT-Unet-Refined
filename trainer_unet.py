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
# from utils import test_single_volume
import torch.nn.functional as F

class KDloss(nn.Module):

    def __init__(self,lambda_x):
        super(KDloss,self).__init__()
        self.lambda_x = lambda_x

    def inter_fd(self,f_s, f_t):
        s_C, t_C, s_H, t_H = f_s.shape[1], f_t.shape[1], f_s.shape[2], f_t.shape[2]
        if s_H > t_H:
            f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
        elif s_H < t_H:
            f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
        else:
            pass
        
        idx_s = random.sample(range(s_C),min(s_C,t_C))
        idx_t = random.sample(range(t_C),min(s_C,t_C))

        #inter_fd_loss = F.mse_loss(f_s[:, 0:min(s_C,t_C), :, :], f_t[:, 0:min(s_C,t_C), :, :].detach())

        inter_fd_loss = F.mse_loss(f_s[:, idx_s, :, :], f_t[:, idx_t, :, :].detach())
        return inter_fd_loss 
    
    def intra_fd(self,f_s):
        sorted_s, indices_s = torch.sort(F.normalize(f_s, p=2, dim=(2,3)).mean([0, 2, 3]), dim=0, descending=True)
        f_s = torch.index_select(f_s, 1, indices_s)
        intra_fd_loss = F.mse_loss(f_s[:, 0:f_s.shape[1]//2, :, :], f_s[:, f_s.shape[1]//2: f_s.shape[1], :, :])
        return intra_fd_loss
    
    def forward(self,feature,feature_decoder,final_up):
        # f1 = feature[0][-1] # 
        # f2 = feature[1][-1]
        # f3 = feature[2][-1]
        # f4 = feature[3][-1] # lower feature 

        f1_0 = feature[0] # 
        f2_0 = feature[1]
        f3_0 = feature[2]
        f4_0 = feature[3] # lower feature 

        # f1_d = feature_decoder[0][-1] # 14 x 14
        # f2_d = feature_decoder[1][-1] # 28 x 28
        # f3_d = feature_decoder[2][-1] # 56 x 56

        f1_d_0 = feature_decoder[0] # 14 x 14
        f2_d_0 = feature_decoder[1] # 28 x 28
        f3_d_0 = feature_decoder[2] # 56 x 56

        #print(f3_d.shape)

        final_layer = final_up
        #print(final_layer.shape)


        # loss =  (self.intra_fd(f1)+self.intra_fd(f2)+self.intra_fd(f3)+self.intra_fd(f4))/4
        loss = (self.intra_fd(f1_0)+self.intra_fd(f2_0)+self.intra_fd(f3_0)+self.intra_fd(f4_0))/4
        loss += (self.intra_fd(f1_d_0)+self.intra_fd(f2_d_0)+self.intra_fd(f3_d_0))/3
        # loss += (self.intra_fd(f1_d)+self.intra_fd(f2_d)+self.intra_fd(f3_d))/3


        
        loss += (self.inter_fd(f1_d_0,final_layer)+self.inter_fd(f2_d_0,final_layer)+self.inter_fd(f3_d_0,final_layer)
                   +self.inter_fd(f1_0,final_layer)+self.inter_fd(f2_0,final_layer)+self.inter_fd(f3_0,final_layer)+self.inter_fd(f4_0,final_layer))/7

        
        
        loss = loss * self.lambda_x
        return loss 
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

import re

def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator, RandomGenerator_DINO
    from torchvision.transforms import functional as VF

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]),
                               transform_dino=transforms.Compose(
                                   [RandomGenerator_DINO(output_size=[args.img_size, args.img_size])]))

    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.001)

    writer = SummaryWriter(snapshot_path + '/log')
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0

    iter_num = 0
    start_epoch = 0

    # === ✅ Resume from checkpoint if specified
    if args.resume:
        if os.path.exists(args.resume):
            checkpoint = torch.load(args.resume)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint.get('epoch', 0) + 1
                logging.info(f"🔁 Resumed training from checkpoint at epoch {start_epoch}")
            else:
                model.load_state_dict(checkpoint)
                epoch_match = re.search(r'epoch_(\d+)', args.resume)
                if epoch_match:
                    start_epoch = int(epoch_match.group(1)) + 1
                else:
                    start_epoch = 0
                logging.info(f"🔁 Loaded model weights only from {args.resume}, resumed from epoch {start_epoch}")
        else:
            raise FileNotFoundError(f"❌ Resume checkpoint not found: {args.resume}")



    momentum_schedule = cosine_scheduler(0.996, 1, max_iterations, len(trainloader))
    iterator = tqdm(range(start_epoch, max_epoch), ncols=70)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
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
            # logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

        # ✅ 保存完整 checkpoint
        if epoch_num > 10 or epoch_num == max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, f'epoch_{epoch_num}.pth')
            torch.save({
                'epoch': epoch_num,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, save_mode_path)
            logging.info("💾 Saved checkpoint to {}".format(save_mode_path))

    writer.close()
    return "Training Finished!"
