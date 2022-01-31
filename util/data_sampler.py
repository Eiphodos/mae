import os
import multiprocessing as mp
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset

class NiftiVolDataset(Dataset):
    def __init__(self, args, nifti_suffix='nii.gz'):
        self.data_folder = args.data_path
        self.input_size = args.input_size
        self.files = os.listdir(self.data_folder)

        # Preprocessing
        self.pp_pipeline = build_pp_pipeline(args)
        self.pp_to_tmp = args.pp_to_temp
        if self.pp_to_tmp:
            self.run_pp_pipeline_all_files()

    def __len__(self):
        return len(self.files)


    def __getitem__(self, item):
        file_path = os.path.join(self.data_folder, self.files[item])
        data = nib.load(file_path)
        if self.input_size < data.shape:
            data = pad_data(data)
            subvolume = self.sample_subvolume(data)
        else:
            subvolume = self.sample_subvolume(data)
        return subvolume
        vol = apply_pp_pipeline(self.pp_pipeline, file_path)


    def sample_subvolume(self, data):
        start_index_x = np.random.randint(0, data.shape[0] - self.input_size)
        start_index_y = np.random.randint(0, data.shape[1] - self.input_size)
        start_index_z = np.random.randint(0, data.shape[2] - self.input_size)

        subvolume = data.slice[start_index_x:self.input_size, start_index_y:self.input_size, start_index_z:self.input_size]
        return subvolume


    def build_pp_pipeline(self, args):
        pp_pipeline = []
        job = ReadNiftiFile(source_folder=self.data_folder)
        pp_pipeline.append(job)
        if args.clip_ct:
            job = ClipCTWindow(cubed=args.cubed_ct, ct_min=args.ct_min, ct_max=args.ct_max)
            pp_pipeline.append()
        if args.interpolate_voxel_spacing:
            job = InterpolateVoxelSpacing(voxel_spacing=args.voxel_spacing)
        if args.copy_to_tmp:
            job = SaveNiftiFile(dest_folder=args.tmp_dir)
            pp_pipeline.append(job)
        return pp_pipeline

    def run_pp_pipeline_all_files(self, args):
        assert args.copy_to_tmp
        for f in self.files:
            self.apply_pp_pipeline(f)



    def apply_pp_pipeline(self, x):
        for pp in self.pp_pipeline:
            x = pp(x)
        return x



