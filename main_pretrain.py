# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import neptune.new as neptune

import torch
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import torchio as tio
from torch.distributed.elastic.multiprocessing.errors import record

import timm
import timm.optim.optim_factory as optim_factory

from data.datasets import build_dataset_pretrain
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from models import models_mae

from engine_pretrain import train_one_epoch

assert timm.__version__ == "0.3.2"  # version check


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--save_ckpt_freq', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    parser.add_argument('--debug', action='store_true',
                        help='Debug mode, slower but more verbose')
    parser.set_defaults(debug=False)

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', nargs='*', default=[224], type=int,
                        help='images input size, can be a single number or for example --input_size 128 64 32 as H*W*D')

    parser.add_argument('--patch_size', nargs='*', default=[16], type=int,
                        help='patch input size, can be a single number or for example --patch_size 128 64 32 as H*W*D')

    parser.add_argument('--sample_size', nargs='*', default=[224], type=int,
                        help='The size to sample from the original volume')

    parser.add_argument('--input_dim', default=2, type=int,
                        help='Dimension of the input, allowed values are 2 and 3')

    parser.add_argument('--channels', default=3, type=int,
                        help='Number of channels for the images')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    parser.add_argument('--mean_patch_loss', action='store_true',
                        help='Add additional an additiona loss term for the mean of each patch')
    parser.set_defaults(mean_patch_loss=False)

    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision for model, operations and input')
    parser.set_defaults(mixed_precision=False)


    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Data transform parameters
    parser.add_argument('--pp_ct_intensity', default=False, action='store_true',
                        help='If input images should be clipped to a set intensity during preprocessing')
    parser.add_argument('--transform_ct_intensity', default=False, action='store_true',
                        help='If input images should be clipped to a set intensity during transform')
    parser.add_argument('--cube_root_ct', default=False, action='store_true',
                        help='If the ct intensity should be cube rooted before scaled into the correct span')
    parser.add_argument('--ct_intensity_min', default=-1000, type=int,
                        help='Minimum CT intensity')
    parser.add_argument('--ct_intensity_max', default=1000, type=int,
                        help='Maximum CT intensity')
    parser.add_argument('--voxel_interpolation', default=False, action='store_true',
                        help='If voxel spacing should be interpolated into a new space')
    parser.add_argument('--voxel_spacing', nargs='*', type=float, default=[1.0],
                        help='The voxel spacing to interpolate to. Can be a single value which then will be used for '
                             'xyz or a tuple of 3 values. Example: --voxel_spacing 1, --voxel_spacing 1 1.5 2')
    parser.add_argument('--norm_mean', default=0.1943, type=float,
                        help='Mean for normalization')
    parser.add_argument('--norm_std', default=0.2786, type=float,
                        help='Standard deviation for normalization')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--datasets', nargs='*', type=str, help='The datasets used in pretraining.')

    parser.add_argument('--use_tmp_dir', default=False, action='store_true',
                        help='If data should be extract from data_path to a local temp directory')
    parser.add_argument('--metadata_file', default='', type=str,
                        help='File containing metadata for pre-processing')

    parser.add_argument('--queue_length', default=1024, type=int,
                        help='The max number of samples in the torchio queue')
    parser.add_argument('--samples_per_volume', default=32, type=int,
                        help='The number of samples generated from each volume in the torchio queue')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


@record
def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # Fix the input size if its a single value
    if len(args.input_size) == 1:
        args.input_size = args.input_size[0]
    else:
        args.input_size = tuple(args.input_size)
    if len(args.sample_size) == 1:
        args.input_size = args.sample_size[0]
    else:
        args.input_size = tuple(args.sample_size)
    if len(args.patch_size) == 1:
        args.patch_size = args.patch_size[0]
    else:
        args.patch_size = tuple(args.patch_size)

    # Build dataset and transform
    dataset_train = build_dataset_pretrain(args)
    print(dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        if args.input_dim == 3:
            sampler_train = tio.UniformSampler(patch_size=args.sample_size)
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
            print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(logdir=args.log_dir)
    else:
        log_writer = None

    if global_rank == 0:
        neptune_logger = neptune.init()
        neptune_logger['parameters'] = vars(args)
        tags = ['Pretraining', 'MAE', 'Official', 'DeepLesion', 'ViT']
        if args.input_dim == 3:
            tags.append('3D')
        else:
            tags.append('2D')
        neptune_logger['sys/tags'].add(tags)

    if args.input_dim == 3:
        patches_queue = tio.Queue(
            dataset_train,
            args.queue_length,
            args.samples_per_volume,
            sampler_train,
            num_workers=args.num_workers
        )

        data_loader_train = torch.utils.data.DataLoader(
            patches_queue,
            batch_size=args.batch_size,
            num_workers=0,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
    else:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

    # define the model
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss,
                                            mean_patch_loss=args.mean_patch_loss,
                                            in_chans=args.channels,
                                            img_size=args.input_size,
                                            patch_size=args.patch_size)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler(enabled=args.mixed_precision)

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and args.input_dim != 3:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch, }

        if misc.is_main_process():
            misc.log_to_neptune(neptune_logger, log_stats)

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if misc.is_main_process():
        neptune_logger.stop()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
