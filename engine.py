# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
import sys
from typing import Iterable

import torch
from einops import rearrange
from tqdm import tqdm

import utils
from losses import HuberLoss
from utils import lab2rgb, psnr, rgb2lab


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16,
                    log_writer=None, lr_scheduler=None, start_steps=None, lr_schedule_values=None,
                    wd_schedule_values=None, exp_name=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    # loss_func = nn.MSELoss()
    loss_func = HuberLoss()

    for step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if step % 100 == 0:
            print(exp_name)
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        images, bool_hinted_pos = batch

        images = images.to(device, non_blocking=True)
        # bool_hinted_pos = bool_hinted_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
        bool_hinted_pos = bool_hinted_pos.to(device, non_blocking=True).to(torch.bool)

        # Lab conversion and normalizatoin
        images = rgb2lab(images, 50, 100, 110)  # l_cent, l_norm, ab_norm
        B, C, H, W = images.shape
        h, w = H // patch_size, W // patch_size

        # import pdb; pdb.set_trace()
        with torch.no_grad():
            # calculate the predict label
            images_patch = rearrange(images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
            labels = rearrange(images_patch, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)

        with torch.cuda.amp.autocast():
            outputs = model(images, bool_hinted_pos)  # ! images has been changed (in-place ops)
            outputs = rearrange(outputs, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)

            # Loss is calculated only with the ab channels
            loss = loss_func(input=outputs, target=labels[:, :, :, 1:])

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def validate(model: torch.nn.Module, data_loader: Iterable, device: torch.device,
             patch_size: int = 16, log_writer=None, val_hint_list=[10]):
    model.eval()
    header = 'Validation'

    psnr_sum = dict(zip(val_hint_list, [0.] * len(val_hint_list)))
    num_validated = 0
    with torch.no_grad():
        for step, (batch, _) in tqdm(enumerate(data_loader), desc=header, ncols=100, total=len(data_loader)):
            # assign learning rate & weight decay for each step
            images, bool_hints = batch
            B, _, H, W = images.shape
            h, w = H // patch_size, W // patch_size

            images = images.to(device, non_blocking=True)
            # Lab conversion and normalizatoin
            images_lab = rgb2lab(images)
            # calculate the predict label
            images_patch = rearrange(images_lab, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
            labels = rearrange(images_patch, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)

            for idx, count in enumerate(val_hint_list):
                bool_hint = bool_hints[:, idx].to(device, non_blocking=True).flatten(1).to(torch.bool)
                # bool_hint = bool_hints.to(device, non_blocking=True).to(torch.bool)

                with torch.cuda.amp.autocast():
                    outputs = model(images_lab.clone(), bool_hint.clone())
                    outputs = rearrange(outputs, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)

                pred_imgs_lab = torch.cat((labels[:, :, :, 0].unsqueeze(3), outputs), dim=3)
                pred_imgs_lab = rearrange(pred_imgs_lab, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)',
                                          h=h, w=w, p1=patch_size, p2=patch_size)
                pred_imgs = lab2rgb(pred_imgs_lab)

                _psnr = psnr(images, pred_imgs) * B
                psnr_sum[count] += _psnr.item()
            num_validated += B

        psnr_avg = dict()
        for count in val_hint_list:
            psnr_avg[f'psnr@{count}'] = psnr_sum[count] / num_validated

        torch.cuda.synchronize()

        if log_writer is not None:
            log_writer.update(head="psnr", **psnr_avg)
    return psnr_avg
