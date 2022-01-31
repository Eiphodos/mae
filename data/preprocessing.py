import numpy as np
import nibabel as nib
import os


class ReadNiftiFile:
    def __init__(self, source_folder):
        self.source_folder = source_folder

    def __call__(self, filename):
        filepath = os.path.join(self.source_folder, filename)
        data = nib.load(filepath)
        return data, filename


class ClipCTIntensity:
    def __init__(self, ct_min=-1000, ct_max=1000, cuberoot_compression=True):
        self.cuberoot_compression = cuberoot_compression
        self.ct_min = ct_min
        self.ct_max = ct_max

    def __call__(self, np_arr):
        if self.cuberoot_compression:
            np_arr = np.clip(np_arr, self.ct_min, self.ct_max)
        else:
            np_arr = np.clip(np_arr, self.ct_min, self.ct_max)
            np_arr = np.cbrt(np_arr)
        return np_arr


class SaveNiftiFile:
    def __init__(self, dest_folder):
        self.dest_folder = dest_folder

    def __call__(self, np_arr):
        data = nib.Nifti1.NiftiImage(np_arr, None)
        out_path = os.path.join(self.dest_folder, )