import torchio as tio
import numpy as np


class TioRandomResizedCropOrPad(tio.SpatialTransform):
    def __init__(
            self,
            target_shape,
            scale=(0.2, 1.0),
            image_interpolation='linear',
            label_interpolation='nearest'):
        self.target_shape = target_shape
        self.scale = scale
        self.image_interpolation = image_interpolation
        self.label_interpolation = label_interpolation

        self.resize_transform = tio.Resize(target_shape, image_interpolation, label_interpolation)

    def apply_transform(self, subject):
        shape_in = np.asarray(subject.spatial_shape)
        random_scale = np.random.uniform(low=self.scale[0], high=self.scale[1])
        crop_shape = (np.ceil(shape_in*random_scale)).astype(np.int32)
        crop = tio.CropOrPad(target_shape=crop_shape)
        cropped_subject = crop(subject)
        resized_subject = self.resize_transform(cropped_subject)

        return resized_subject