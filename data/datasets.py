# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
from PIL import Image
import numpy as np
import pandas as pd
import multiprocessing as mp
from itertools import repeat
import torchio as tio
import nibabel as nib

from torchvision import datasets, transforms
import torch.distributed as dist

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from util.misc import is_main_process, is_dist_avail_and_initialized
from data.preprocessing import RescaleIntensityCubeRoot, ZNormalizationFixed
from data.transforms import TioRandomResizedCropOrPad

NPZ_SUFFIX = '.npz'
NIFTI_SUFFIX = '.nii.gz'


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
        transforms.Resize(size, interpolation='bicubic'),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def build_dataset_pretrain(args):
    transform = build_transform_pretrain(args)
    if args.input_dim == 3:
        dataset = build_tio_dataset(args, transform)
        return dataset
    else:
        if args.use_tmp_dir:
            img_folder = os.getenv('TMPDIR')
            assert img_folder is not None
            if is_main_process():
                print("Extracting dataset to {} on rank {}".format(img_folder, args.rank))
                extract_dataset_to_local(args.data_path, img_folder, args.metadata_file,
                                         args.pp_ct_intensity, args.ct_intensity_min, args.ct_intensity_max)
                files = os.listdir(img_folder)
                print("Finished extracting {} files in dataset on rank {}".format(len(files), args.rank))
            if is_dist_avail_and_initialized():
                dist.barrier(args.rank)
        else:
            img_folder = args.data_path

        if args.channels == 1:
            loader = grayscale_loader
        elif args.channels == 3:
            loader = rgb_loader
        else:
            raise ValueError('Only Images with either 1 or 3 channels are supported')
        return datasets.folder.ImageFolder(img_folder, loader=loader, transform=transform)


def build_tio_dataset(args, transform):
    files = os.listdir(args.data_path)
    print("Preparing torchio dataset using {} files".format(len(files)))
    subjects_list = []
    for f in files:
        fp = os.path.join(args.data_path, f)
        subject = tio.Subject(t1=tio.ScalarImage(fp))
        org_shape = list(subject.shape)[1:]
        if args.voxel_interpolation:
            affine = subject['t1'].affine[np.nonzero(subject['t1'].affine)][:-1]
            shape = np.array(np.ceil(org_shape * abs(affine) / args.voxel_spacing))
        else:
            shape = np.array(org_shape)
        if (shape >= args.input_size).all():
            subjects_list.append(subject)
    print('Number of files in resulting subject list: {}'.format(len(subjects_list)))

    subjects_dataset = tio.SubjectsDataset(subjects_list, transform=transform)
    return subjects_dataset


def build_transform_pretrain(args):
    if args.channels == 3:
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
    else:
        # DeepLesion mean and std
        mean = 0.1923
        std = 0.2757
    if args.input_dim == 3:
        custom_t = []
        default_t = [
            TioRandomResizedCropOrPad(args.input_size, scale=(0.2, 1.0)),
            tio.RandomAffine(degrees=0),  # Only random scaling, no rotation.
            tio.RandomFlip(axes=(0, 1)),
            ZNormalizationFixed(mean, std)
        ]
        if args.voxel_interpolation:
            custom_t.append(tio.Resample(args.voxel_spacing[0] if len(args.voxel_spacing) < 2 else tuple(args.voxel_spacing)))
        if args.transform_ct_intensity:
            custom_t.append(RescaleIntensityCubeRoot(out_min_max=(0, 1),
                                                     in_min_max=(args.ct_intensity_min, args.ct_intensity_max),
                                                     cube_rooted=True))
        else:
            custom_t.append(RescaleIntensityCubeRoot(out_min_max=(0, 1),
                                                     cube_rooted=False))
        t = tio.Compose(custom_t + default_t)
    else:
        custom_t = []
        default_t = [
                transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)]
        if args.transform_ct_intensity:
            custom_t.append(ClipCTIntensity(args.ct_intensity_min, args.ct_intensity_max))
        t = transforms.Compose(custom_t + default_t)
    return t


def grayscale_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')


def rgb_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def extract_dataset_to_local(root, image_folder, metadata_file, pp_ct, ct_min, ct_max):
    root, dirs, files = next(os.walk(root))
    os.makedirs(image_folder, exist_ok=True)
    nprocs = mp.cpu_count()
    pool = mp.Pool(processes=nprocs)
    if pp_ct and (metadata_file is not ''):
        metadata = pd.read_csv(metadata_file)
        file_ct_min = []
        file_ct_max = []
        for fn in files:
            try:
                pat_idx, study_idx, series_id = int(fn[0:6].lstrip('0')), int(fn[7:9].lstrip('0')), int(fn[10:12].lstrip('0'))
                row = metadata[(metadata['Patient_index'] == pat_idx) & (metadata['Study_index'] == study_idx) & (
                            metadata['Series_ID'] == series_id)]
                dcm_w = row['DICOM_windows'].iloc[0]
                n_ct_min, n_ct_max = int(float(dcm_w.split(', ')[0])), int(float(dcm_w.split(', ')[1]))
                file_ct_min.append(n_ct_min)
                file_ct_max.append(n_ct_max)
            except IndexError as ie:
                print(
                    'Failed to read row for file {} with pat_idx {}, study_idx {}, series_id {} and error {}'.format(
                        fn, pat_idx, study_idx, series_id, ie))
                file_ct_min.append(ct_min)
                file_ct_min.append(ct_max)
        pool.starmap(extract_nifti_to_disk, zip(files, repeat(root), repeat(image_folder),
                                                repeat(pp_ct), file_ct_min, file_ct_max))
    else:
        pool.starmap(extract_nifti_to_disk, zip(files, repeat(root), repeat(image_folder),
                                                repeat(pp_ct), repeat(ct_min), repeat(ct_max)))
    pool.close()
    pool.join()


def extract_nifti_to_disk(file, root, image_folder, pp_ct, ct_min, ct_max):
    try:
        fn = file[:-len(NIFTI_SUFFIX)]
        case_folder = os.path.join(image_folder, fn)
        os.makedirs(case_folder, exist_ok=True)
        data = nib.load(os.path.join(root, file))
        np_data = data.get_fdata()
        if pp_ct:
            np_data = clip_ct_window_cube_root(np_data, ct_min, ct_max)
        np_data = np.transpose(np_data, (2, 0, 1))
        for i, v_slice in enumerate(np_data):
            filename = fn + '_' + str(i) + '.png'
            im = Image.fromarray(v_slice)
            im = im.convert('L')
            im.save(os.path.join(case_folder, filename))
    except Exception as e:
        print('Failed to processes file {} with error {}'.format(file, e))


def extract_npz_to_disk(file, root, image_folder, pp_ct, ct_min, ct_max):
    try:
        fn = file[:-len(NPZ_SUFFIX)]
        case_folder = os.path.join(image_folder, fn)
        os.makedirs(case_folder, exist_ok=True)
        data = np.load(os.path.join(root, file))
        for i, arr in enumerate(data):
            if pp_ct:
                arr = clip_ct_window(arr, ct_min, ct_max)
            filename = fn + '_' + str(i) + '.png'
            im = Image.fromarray(data[arr])
            im.save(os.path.join(case_folder, filename))
    except Exception as e:
        print('Failed to processes file {} with error {}'.format(file, e))


def clip_ct_window(np_arr, ct_min, ct_max):
    np_arr = np.minimum(255, np.maximum(0, (np_arr - ct_min) / (ct_max - ct_min) * 255))
    return np_arr.astype(np.uint8)


def clip_ct_window_cube_root(np_arr, ct_min, ct_max):
    np_arr = np.clip(np_arr, ct_min, ct_max)
    np_arr = np.cbrt(np_arr)
    np_min = np.cbrt(ct_min)
    np_max = np.cbrt(ct_max)
    np_arr = np.minimum(255, np.maximum(0, (np_arr - np_min) / (np_max - np_min) * 255))
    return np_arr.astype(np.uint8)


class ClipCTIntensity:
    def __init__(self, ct_min, ct_max):
        self.ct_min = ct_min
        self.ct_max = ct_max

    def __call__(self, img):
        npimg = np.array(img).astype(np.int32)
        windowed_npimg = np.minimum(255, np.maximum(0, (npimg-self.ct_min)/(self.ct_max-self.ct_min)*255))
        windowed_npimg = windowed_npimg.astype(np.uint8)
        windowed_img = Image.fromarray(windowed_npimg)
        return windowed_img.convert('L')

    def __repr__(self):
        return self.__class__.__name__ + '(min={}, max={})'.format(self.ct_min, self.ct_max)