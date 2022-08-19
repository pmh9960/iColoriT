import argparse
import inspect
import os
import os.path as osp
import random
import sys

import lpips
import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, ToTensor
from tqdm import tqdm

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from utils import psnr


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', type=str, default='data/pred', help='prediction images directory')
    parser.add_argument('--gt_dir', type=str, default='data/gt', help='ground truth images directory')
    parser.add_argument('--hint_size', type=int, default=2)
    parser.add_argument('--num_hint', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--gray_file_list_txt', type=str, default='')
    args = parser.parse_args()

    args.pred_dir = osp.join(args.pred_dir, f'h{args.hint_size}-n{args.num_hint}')
    assert osp.isdir(args.pred_dir), f'{args.pred_dir} is not exists'
    args.save_path = osp.join(args.save_dir, f'h{args.hint_size}-n{args.num_hint}.txt')
    os.makedirs(osp.dirname(args.save_path), exist_ok=True)

    return args


class GtPredImageDataset(Dataset):
    def __init__(self, gt_dir, pred_dir, gray_file_list_txt='') -> None:
        super().__init__()
        self.gt_dir = gt_dir
        self.pred_dir = pred_dir

        # do not use gray images
        self.gray_imgs = []
        if gray_file_list_txt:
            with open(gray_file_list_txt, 'r') as f:
                self.gray_imgs = [osp.splitext(osp.basename(i))[0] for i in f.readlines()]

        gt_files = sorted(os.listdir(self.gt_dir))
        gt_files = [i for i in gt_files if not self.is_gray(i)]
        pred_files = sorted(os.listdir(self.pred_dir))
        pred_files = [i for i in pred_files if not self.is_gray(i)]

        assert len(gt_files) == len(pred_files), f'{len(gt_files)} != {len(pred_files)}'
        for gt_file, pred_file in zip(gt_files, pred_files):
            assert osp.splitext(gt_file)[0] == osp.splitext(pred_file)[0], f'{gt_file} != {pred_file}'

        self.gt_files = gt_files
        self.pred_files = pred_files

        self.tf = Compose([
            Resize((224, 224)),
            ToTensor()
        ])

    def is_gray(self, file):
        return osp.splitext(file)[0] in self.gray_imgs

    def __len__(self):
        return len(self.gt_files)

    def __getitem__(self, idx):
        gt = Image.open(osp.join(self.gt_dir, self.gt_files[idx])).convert('RGB')
        pred = Image.open(osp.join(self.pred_dir, self.pred_files[idx])).convert('RGB')
        gt = self.tf(gt)
        pred = self.tf(pred)
        return (gt, pred), osp.splitext(self.gt_files[idx])[0]


class AverageMeter:
    def __init__(self, **kwargs) -> None:
        self.logger = kwargs
        self.total = 0

    def update(self, n=1, **kwargs):
        for name, value in kwargs.items():
            self.logger[name] += value * n

    def step(self, n=1):
        self.total += n

    def get_avg(self):
        avg = dict()
        for name, value in self.logger.items():
            avg[name] = value / self.total
        return avg

    def __repr__(self) -> str:
        line = ''
        for name, value in self.get_avg().items():
            line += f'{name}: {value:.2f}, '
        return line[:-2]


def calc_psnr(preds, gts):
    return psnr(preds, gts).item()


def make_boundary_mask(patch_size, h, w) -> torch.Tensor:
    assert h % patch_size == 0 and w % patch_size == 0
    mask = torch.zeros((h, w))
    for i in range(h // patch_size - 1):
        mask[(i + 1) * patch_size - 1] = 1
        mask[(i + 1) * patch_size] = 1
    for j in range(w // patch_size - 1):
        mask[:, (j + 1) * patch_size - 1] = 1
        mask[:, (j + 1) * patch_size] = 1
    return mask


def calc_boundary_psnr(img1: torch.Tensor, img2: torch.Tensor, epsilon=1e-5, patch_size=16) -> float:
    '''
    Peak Signal to Noise Ratio along the patches boundary (B-PSNR)
    https://arxiv.org/abs/2207.06831
    '''
    assert img1.dim() == img2.dim()
    h, w = img1.shape[-2], img1.shape[-1]

    mask = make_boundary_mask(patch_size, h, w)
    assert mask.sum() > 0

    if img1.dim() == 4:
        mse = torch.mean((img1 - img2) ** 2, 1)
        mse = mse * mask
        mse = mse.sum((-1, -2)) / mask.sum()
        mse[mse <= epsilon] = epsilon
    else:
        mse = (img1 - img2) ** 2
        mse = mse * mask
        mse = mse.sum() / mask.sum()
        mse = epsilon if mse <= epsilon else mse
    PIXEL_MAX = 1
    psnrs = 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))
    return psnrs.mean().item()


def calc_patch_error_variance(img1: torch.Tensor, img2: torch.Tensor, patch_size=16) -> float:
    '''
    Patch Error Variance (PEV)
    https://arxiv.org/abs/2207.06831
    '''
    assert img1.dim() == img2.dim()
    if img1.dim() == 3 and img2.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    if img1.dim() == 4 and img2.dim() == 4:
        mse = torch.mean((img1 * 255 - img2 * 255) ** 2, 1)
        mse = rearrange(mse, 'b (h p1) (w p2) -> b (h w) (p1 p2)', p1=patch_size, p2=patch_size)
        mse = torch.sqrt(mse.mean(2))
    else:
        raise NotImplementedError(img1, img2.shape)
    return torch.mean(mse.var(1)).item()


def evaluate_from_images(args):
    torch.manual_seed(4885)
    np.random.seed = 4885
    random.seed(4885)

    dataset = GtPredImageDataset(args.gt_dir, args.pred_dir, args.gray_file_list_txt)
    dataloader = DataLoader(dataset, batch_size=16, num_workers=8, drop_last=False, shuffle=False)
    avgmeter = AverageMeter(psnr=0., lpips=0., boundary_psnr=0., pev=0.)
    loss_fn_vgg = lpips.LPIPS(net='vgg').to('cuda')

    pbar = tqdm(desc='Evaluation', total=len(dataloader))
    for batch in dataloader:
        (gts, preds), names = batch
        B = gts.size(0)
        cur_psnr = calc_psnr(preds, gts)
        cur_boundary_psnr = calc_boundary_psnr(preds, gts, patch_size=16)
        cur_pev = calc_patch_error_variance(preds, gts, patch_size=16)
        with torch.no_grad():
            cur_lpips = loss_fn_vgg(preds.to('cuda'), gts.to('cuda')).mean().cpu().item()

        avgmeter.update(psnr=cur_psnr, lpips=cur_lpips,
                        boundary_psnr=cur_boundary_psnr, pev=cur_pev, n=B)
        avgmeter.step(B)
        pbar.set_postfix_str(str(avgmeter))
        pbar.update()
    pbar.close()

    with open(args.save_path, 'w') as f:
        f.write(f'total shown: {avgmeter.total}\n')
        f.write(str(avgmeter.get_avg()))


if __name__ == '__main__':
    args = get_args()
    evaluate_from_images(args)
