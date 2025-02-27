import numpy as np
import nibabel as nib
import os
import warnings
from typing import Optional, Tuple

import torch
import numpy as np

from torchio.data.subject import Subject
from torchio.typing import TypeRangeFloat
from torchio.transforms.preprocessing.intensity.normalization_transform import NormalizationTransform, TypeMaskingMethod


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


class ZNormalizationFixed(NormalizationTransform):
    """Subtract mean and divide by standard deviation.

    Args:
        masking_method: See
            :class:`~torchio.transforms.preprocessing.intensity.NormalizationTransform`.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.
    """
    def __init__(
            self,
            mean,
            std,
            masking_method: TypeMaskingMethod = None,
            **kwargs
            ):
        super().__init__(masking_method=masking_method, **kwargs)
        self.args_names = ('masking_method',)
        self.mean = mean
        self.std = std

    def apply_normalization(
            self,
            subject: Subject,
            image_name: str,
            mask: torch.Tensor,
            ) -> None:
        image = subject[image_name]
        standardized = self.znorm(image.data)
        if standardized is None:
            message = (
                'Standard deviation is 0 for masked values'
                f' in image "{image_name}" ({image.path})'
            )
            raise RuntimeError(message)
        image.set_data(standardized)

    def znorm(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.clone().float()
        if self.std == 0:
            return None
        tensor -= self.mean
        tensor /= self.std
        return tensor


class RescaleIntensityCubeRoot(NormalizationTransform):
    """Rescale intensity values to a certain range.

    Args:
        out_min_max: Range :math:`(n_{min}, n_{max})` of output intensities.
            If only one value :math:`d` is provided,
            :math:`(n_{min}, n_{max}) = (-d, d)`.
        percentiles: Percentile values of the input image that will be mapped
            to :math:`(n_{min}, n_{max})`. They can be used for contrast
            stretching, as in `this scikit-image example`_. For example,
            Isensee et al. use ``(0.5, 99.5)`` in their `nn-UNet paper`_.
            If only one value :math:`d` is provided,
            :math:`(n_{min}, n_{max}) = (0, d)`.
        masking_method: See
            :class:`~torchio.transforms.preprocessing.intensity.NormalizationTransform`.
        in_min_max: Range :math:`(m_{min}, m_{max})` of input intensities that
            will be mapped to :math:`(n_{min}, n_{max})`. If ``None``, the
            minimum and maximum input intensities will be used.
        cube_rooted: Boolean that controls if array (and in_min_max) should be cube_rooted before use)
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.
    """  # noqa: E501
    def __init__(
            self,
            out_min_max: TypeRangeFloat = (0, 1),
            percentiles: TypeRangeFloat = (0, 100),
            masking_method: TypeMaskingMethod = None,
            in_min_max: Optional[Tuple[float, float]] = None,
            cube_rooted=True,
            **kwargs
            ):
        super().__init__(masking_method=masking_method, **kwargs)
        self.out_min_max = out_min_max
        self.in_min_max = in_min_max
        self.out_min, self.out_max = self._parse_range(
            out_min_max, 'out_min_max')
        self.percentiles = self._parse_range(
            percentiles, 'percentiles', min_constraint=0, max_constraint=100)
        self.args_names = 'out_min_max', 'percentiles', 'masking_method'
        self.cube_rooted = cube_rooted
        if self.cube_rooted:
            self.in_min_max = np.cbrt(self.in_min_max)

    def apply_normalization(
            self,
            subject: Subject,
            image_name: str,
            mask: torch.Tensor,
            ) -> None:
        image = subject[image_name]
        image.set_data(self.rescale(image.data, mask, image_name))

    def rescale(
            self,
            tensor: torch.Tensor,
            mask: torch.Tensor,
            image_name: str,
            ) -> torch.Tensor:
        # The tensor is cloned as in-place operations will be used
        array = tensor.clone().float().numpy()
        mask = mask.numpy()
        if not mask.any():
            message = (
                f'Rescaling image "{image_name}" not possible'
                ' because the mask to compute the statistics is empty'
            )
            warnings.warn(message, RuntimeWarning)
            return tensor
        values = array[mask]
        cutoff = np.percentile(values, self.percentiles)
        np.clip(array, *cutoff, out=array)
        if self.cube_rooted:
            array = np.cbrt(array)
        if self.in_min_max is None:
            in_min, in_max = array.min(), array.max()
        else:
            in_min, in_max = self.in_min_max
        in_range = in_max - in_min
        if in_range == 0:  # should this be compared using a tolerance?
            message = (
                f'Rescaling image "{image_name}" not possible'
                ' because all the intensity values are the same'
            )
            warnings.warn(message, RuntimeWarning)
            return tensor
        array -= in_min
        array /= in_range
        out_range = self.out_max - self.out_min
        array *= out_range
        array += self.out_min
        return torch.as_tensor(array)
