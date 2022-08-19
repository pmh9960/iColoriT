# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'

import argparse
import datetime
import json
import os
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from timm.models import create_model
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

import modeling  # To register models
import utils
from datasets import build_pretraining_dataset, build_fixed_validation_dataset, build_validation_dataset
from engine import train_one_epoch, validate
from optim_factory import create_optimizer
from utils import NativeScalerWithGradNormCount as NativeScaler


def get_args():
    parser = argparse.ArgumentParser('iColoriT training scripts', add_help=False)
    # Training
    parser.add_argument('--exp_name', default='', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--save_ckpt_freq', default=5, type=int)
    parser.add_argument('--seed', default=4885, type=int)
    parser.add_argument('--save_args_pkl', action='store_true', help='Save args as pickle file')
    parser.add_argument('--no_save_args_pkl', action='store_false', dest='save_args_pkl', help='')
    parser.set_defaults(save_args_pkl=True)
    parser.add_argument('--save_args_txt', action='store_true', help='Save args as txt file')

    # Dataset
    parser.add_argument('--input_size', default=224, type=int, help='images input size for backbone')
    parser.add_argument('--data_path', default='data/train/images', type=str, help='dataset path')
    parser.add_argument('--val_data_path', default='data/val/images', type=str, help='validation dataset path')
    parser.add_argument('--val_hint_dir', type=str, help='hint directory for fixed validation', default='data/hint')
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='tf_log', help='path where to tensorboard log')
    parser.add_argument('--resume', default='', help='path of checkpoint directory (force_resume should be True)')
    parser.add_argument('--force_resume', action='store_true')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch (changed by resume function if needed)')
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem', help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--gray_file_list_txt', type=str, default='', help='use gray file list to exclude them')
    parser.add_argument('--return_name', action='store_true', help='return name for saving (False for train)')

    # Model
    parser.add_argument('--model', default='icolorit_base_4ch_patch16_224', type=str, help='Name of model to train')
    parser.add_argument('--use_rpb', action='store_true', help='relative positional bias')
    parser.add_argument('--no_use_rpb', action='store_false', dest='use_rpb')
    parser.set_defaults(use_rpb=True)
    parser.add_argument('--head_mode', type=str, default='cnn', help='head_mode', choices=['linear', 'cnn', 'locattn'])
    parser.add_argument('--drop_path', type=float, default=0.0, help='Drop path rate')
    parser.add_argument('--mask_cent', action='store_true', help='mask_cent')

    # Hint Generator
    parser.add_argument('--hint_generator', type=str, default='RandomHintGenerator')
    parser.add_argument('--num_hint_range', default=[0, 128], type=int, nargs=2, help='# hints range for each image')
    parser.add_argument('--hint_size', default=2, type=int, help='size of the hint region is given by (h, h)')
    parser.add_argument('--avg_hint', action='store_true', help='avg hint')
    parser.add_argument('--no_avg_hint', action='store_false', dest='avg_hint')
    parser.set_defaults(avg_hint=True)
    parser.add_argument('--val_hint_list', default=[1, 10, 100], nargs='+')

    # Learning rate scheduling
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, help='warmup learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=3, help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, help='steps to warmup LR, priority: steps -> epochs')
    parser.add_argument('--min_lr', type=float, default=1e-5,
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    # Optimizer
    parser.add_argument('--opt', default='adamw', type=str, help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=(0.9, 0.95), type=float, nargs='+',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""")

    # distributed training parameters
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)
    return parser.parse_args()


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
        mask_cent=args.mask_cent,
    )
    return model


def main(args):
    utils.init_distributed_mode(args)
    # print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    model = get_model(args)
    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    # get dataset
    dataset_train = build_pretraining_dataset(args)
    dataset_val = build_fixed_validation_dataset(args)
    # dataset_val = build_validation_dataset(args)  # validate without fixed hint set

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_rank = global_rank
        num_training_steps_per_epoch = len(dataset_train) // args.batch_size // num_tasks

        sampler_train = DistributedSampler(dataset_train, num_replicas=num_tasks, rank=sampler_rank, shuffle=True)
        sampler_val = DistributedSampler(dataset_val, num_replicas=num_tasks, rank=sampler_rank, shuffle=False)
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = RandomSampler(dataset_train)
        sampler_val = RandomSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None:
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        worker_init_fn=utils.seed_worker
    )
    data_loader_val = DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        worker_init_fn=utils.seed_worker
    )

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params: {} M'.format(n_parameters / 1e6))

    total_batch_size = args.batch_size * utils.get_world_size()
    # args.lr = args.lr * total_batch_size / 256

    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        # For debugging
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps)
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    utils.auto_load_model(args=args, model=model, model_without_ddp=model_without_ddp,
                          optimizer=optimizer, loss_scaler=loss_scaler)
    utils.save_args(args, args.output_dir, save_pkl=args.save_args_pkl, save_txt=args.save_args_txt)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)

        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            patch_size=patch_size[0],
            exp_name=args.exp_name,
        )
        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                val_stats = validate(
                    model, data_loader_val, device, patch_size[0], log_writer,
                    args.val_hint_list,
                )
                utils.save_model(args=args, model=model, model_without_ddp=model_without_ddp,
                                 optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch, 'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    args = get_args()
    if not args.force_resume:
        strtime = time.strftime("%y%m%d_%H%M%S")
        args.exp_name = '_'.join([args.exp_name, strtime])
        args.output_dir = os.path.join(args.output_dir, args.model, args.exp_name)
        args.log_dir = os.path.join(args.log_dir, args.exp_name)
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
    args.hint_dirs = [os.path.join(args.val_hint_dir, f'h{args.hint_size}-n{val_num_hint}')
                      for val_num_hint in args.val_hint_list]
    main(args)
