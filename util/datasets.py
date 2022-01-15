# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL
import numpy as np
import multiprocessing as mp
from itertools import repeat

from torchvision import datasets, transforms
import torch.distributed as dist

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from .misc import is_main_process, is_dist_avail_and_initialized


def build_dataset_finetune(is_train, args):
    transform = build_transform_finetune(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform_finetune(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def build_dataset_pretrain(args):
    transform = build_transform_pretrain(args)

    if args.use_tmp_dir:
        img_folder = os.getenv('TMPDIR')
        assert img_folder is not None
        if is_main_process():
            extract_dataset_to_local(args.data_path, img_folder)
        if is_dist_avail_and_initialized():
            dist.barrier()
    else:
        img_folder = args.data_path

    if args.channels == 1:
        loader = grayscale_loader
    elif args.channels == 3:
        loader = rgb_loader
    else:
        raise ValueError('Only Images with either 1 or 3 channels are supported')

    return datasets.folder.ImageFolder(img_folder, loader=loader, transform=transform)


def build_transform_pretrain(args):
    if args.channels == 3:
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
    else:
        mean = 0
        std = 1

    custom_t = []
    default_t = [
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)]
    if args.clip_ct_intensity:
        custom_t.append(ClipCTIntensity(args.ct_intensity_min, args.ct_intensity_max))
    t = transforms.Compose(custom_t + default_t)
    return t


def grayscale_loader(path):
    with open(path, 'rb') as f:
        img = PIL.Image.open(f)
        nimg = np.array(img)
        nimg = nimg.astype(np.uint8)
        img = PIL.Image.fromarray(nimg)
        return img.convert('L')


def rgb_loader(path):
    with open(path, 'rb') as f:
        img = PIL.Image.open(f)
        return img.convert('RGB')


def extract_dataset_to_local(root, image_folder):
    root, dirs, files = next(os.walk(root))
    nprocs = mp.cpu_count()
    pool = mp.Pool(processes=nprocs)
    pool.starmap(extract_npz_to_disk, zip(files, repeat(root), repeat(image_folder)))
    pool.close()
    pool.join()


def extract_npz_to_disk(file, root, image_folder):
    case_folder = os.path.join(image_folder, file[:-4])
    os.makedirs(case_folder, exist_ok=True)
    data = np.load(os.path.join(root, file))
    for i, arr in enumerate(data):
        filename = file[:-4] + '_' + str(i) + '.png'
        im = PIL.Image.fromarray(data[arr])
        im.save(os.path.join(case_folder, filename))


class ClipCTIntensity:
    def __init__(self, ct_min, ct_max):
        self.ct_min = ct_min
        self.ct_max = ct_max

    def __call__(self, img):
        npimg = np.array(img).astype(np.int32)
        # Convert from 16-bit image not already done.
        if np.min(npimg) > 255:
            npimg = npimg - 32768
        windowed_npimg = np.minimum(255, np.maximum(0, (npimg-self.ct_min)/(self.ct_max-self.ct_min)*255))
        windowed_npimg = windowed_npimg.astype(np.uint8)
        windowed_img = PIL.Image.fromarray(windowed_npimg)
        return windowed_img.convert('L')

    def __repr__(self):
        return self.__class__.__name__ + '(min={}, max={})'.format(self.ct_min, self.ct_max)