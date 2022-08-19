# -*- coding: utf-8 -*-
# @Time    : 2021/11/18 22:40
# @Author  : zhao pengfei
# @Email   : zsonghuan@gmail.com
# @File    : run_mae_vis.py
# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import argparse
import inspect
import math
import os
import os.path as osp
import pickle
import random
import sys
import time
from glob import glob

import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision
from einops import rearrange
from PIL import Image
from timm.models import create_model
from torchvision.transforms import ToTensor
from torchvision.utils import save_image

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import modeling
from datasets import DataTransformationForIColoriT
from utils import lab2rgb, psnr, rgb2lab

from evaluation.rollout import VITAttentionRollout


def get_args():
    parser = argparse.ArgumentParser('Rollout', add_help=False)
    # For Rollout
    parser.add_argument('--img_path', type=str, default='docs/image_for_rollout.JPEG')
    parser.add_argument('--save_dir', type=str, help='save image directory', default='results/rollout')
    parser.add_argument('--model_path', type=str, help='checkpoint path of model', default='checkpoint.pth')
    parser.add_argument('--model_args_path', type=str, help='args.pkl path of model', default='')

    # Dataset parameters
    parser.add_argument('--input_size', default=224, type=int, help='images input size for backbone')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem', help='')
    parser.set_defaults(pin_mem=True)

    # Model parameters
    parser.add_argument('--model', default='icolorit_base_4ch_patch16_224', type=str, help='Name of model to inference')
    parser.add_argument('--use_rpb', action='store_true', help='relative positional bias')
    parser.add_argument('--no_use_rpb', action='store_false', dest='use_rpb')
    parser.set_defaults(use_rpb=True)
    parser.add_argument('--head_mode', type=str, default='cnn', help='head_mode', choices=['linear', 'cnn', 'locattn'])
    parser.add_argument('--drop_path', type=float, default=0.0, help='Drop path rate')
    parser.add_argument('--mask_cent', action='store_true', help='mask_cent')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')

    # Hint generator parameter
    parser.add_argument('--hint_generator', type=str, default='InteractiveHintGenerator')
    parser.add_argument('--hint_size', default=2, type=int, help='size of the hint region is given by (h, h)')
    parser.add_argument('--avg_hint', action='store_true', help='avg hint')
    parser.add_argument('--no_avg_hint', action='store_false', dest='avg_hint')
    parser.set_defaults(avg_hint=True)

    args = parser.parse_args()

    if osp.isdir(args.model_path):
        all_checkpoints = glob(osp.join(args.model_path, 'checkpoint-*.pth'))
        latest_ckpt = -1
        for ckpt in all_checkpoints:
            t = ckpt.split('-')[-1].split('.')[0]
            if t.isdigit():
                latest_ckpt = max(int(t), latest_ckpt)
        if latest_ckpt >= 0:
            args.model_path = os.path.join(args.model_path, 'checkpoint-%d.pth' % latest_ckpt)
    print(f'Load checkpoint: {args.model_path}')

    if args.model_args_path:
        with open(args.model_args_path, 'rb') as f:
            train_args = vars(pickle.load(f))
            model_keys = ['model', 'use_rpb', 'head_mode', 'drop_path', 'mask_cent', 'avg_hint']
            for key in model_keys:
                if key in train_args.keys():
                    setattr(args, key, train_args[key])
                else:
                    print(f'{key} is not in {args.model_args_path}. Please check the args.pkl')
            time.sleep(3)
    print(f'Load args: {args.model_args_path}')

    return args


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        use_rpb=args.use_rpb,
        avg_hint=args.avg_hint,
        head_mode=args.head_mode,
    )
    return model


def main(args):
    # print(args)

    device = torch.device(args.device)
    cudnn.benchmark = True

    model = get_model(args)
    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    model.to(device)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    rollout = VITAttentionRollout(model, attention_layer_name='attn_drop',
                                  head_fusion='mean', discard_ratio=0.9, patch_size=16)

    image = Image.open(args.img_path)
    image.convert('RGB')
    print("img path:", args.img_path)

    tf = DataTransformationForIColoriT(args)

    num_points = 0
    while True:
        img, (bool_hinted_pos, hint_coords) = tf(image)
        bool_hinted_pos = torch.from_numpy(bool_hinted_pos)

        with torch.no_grad():
            img = img.unsqueeze(0).to(device, non_blocking=True)
            bool_hinted_pos = bool_hinted_pos.unsqueeze(0).to(device, non_blocking=True).flatten(1).to(torch.bool)

            # Lab conversion and normalizatoin
            img_lab = rgb2lab(img.clone(), 50, 100, 110)  # l_cent, l_norm, ab_norm
            B, C, H, W = img_lab.shape
            _, L = bool_hinted_pos.shape
            hint_size = int(math.sqrt(H * W // L))  # assume square inputs
            h, w = H // patch_size[0], W // patch_size[1]

            bool_hinted_pos = bool_hinted_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
            rollout_masks, outputs = rollout(img_lab.clone(), bool_hinted_pos, img, hint_coords)

            # show prediction
            # b, (h, w) (p1, p2)
            images_patch = rearrange(img_lab.clone(), 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                                     p1=patch_size[0], p2=patch_size[1])
            l_channel = rearrange(images_patch, 'b n (p1 p2 c) -> b n (p1 p2) c',
                                  p1=patch_size[0], p2=patch_size[1])[:, :, :, 0]
            predicted_ab = rearrange(outputs, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size[0], p2=patch_size[1])
            # b, (h, w), (p1, p2), c
            predicted_lab = torch.cat((l_channel.unsqueeze(3), predicted_ab), dim=3)
            predicted_lab = rearrange(predicted_lab, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)',
                                      h=h, w=w, p1=patch_size[0], p2=patch_size[1])

            # visualize hint
            gray_image = img.clone()
            gray_image = rgb2lab(gray_image)
            _device = '.cuda' if bool_hinted_pos.device.type == 'cuda' else ''
            hint = torch.reshape(bool_hinted_pos, (B, H // hint_size, W // hint_size))
            _hint = hint.unsqueeze(1).type(f'torch{_device}.FloatTensor')
            _full_hint = nn.functional.interpolate(_hint, scale_factor=hint_size)  # Needs to be Double
            full_hint = _full_hint.type(f'torch{_device}.BoolTensor')
            if args.avg_hint:
                _avg_x = nn.functional.interpolate(gray_image, size=(H // hint_size, W // hint_size), mode='bilinear')
                _avg_x[:, 1, :, :].hinted_fill_(hint.squeeze(1), 0)  # 1
                _avg_x[:, 2, :, :].hinted_fill_(hint.squeeze(1), 0)
                x_ab = nn.functional.interpolate(_avg_x, scale_factor=hint_size, mode='nearest')[:, 1:, :, :]
                gray_image = torch.cat((gray_image[:, 0, :, :].unsqueeze(1), x_ab), dim=1)
            else:
                gray_image[:, 1, :, :].hinted_fill_(full_hint.squeeze(1), 0)  # 1
                gray_image[:, 2, :, :].hinted_fill_(full_hint.squeeze(1), 0)
            gray_image = lab2rgb(gray_image)

            # save reconstructed image
            pred_img = lab2rgb(predicted_lab, 50, 100, 110)

            cur_psnr = psnr(img, pred_img)
            print(f'PSNR: {cur_psnr.item():.2f}')

            os.makedirs(args.save_dir, exist_ok=True)

            rollout_masks = [ToTensor()(rollout_mask).cpu().unsqueeze(0) for rollout_mask in rollout_masks] \
                if num_points > 0 else []
            save_image(torch.cat([gray_image.cpu(), pred_img.cpu(), *rollout_masks]),
                       osp.join(args.save_dir, f'rollout_{num_points}.png'))
        num_points += 1


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    opts = get_args()
    main(opts)
