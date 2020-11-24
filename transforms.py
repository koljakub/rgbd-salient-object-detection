# Transformation functions
from copy import deepcopy

import torchvision.transforms.functional as F


class ToTensor:

    def __init__(self, apply_to=['img_rgb', 'img_depth', 'img_gt']):
        self.apply_to = apply_to

    def __call__(self, sample):
        sample_t = deepcopy(sample)

        for img_type in self.apply_to:
            setattr(sample_t, img_type, F.to_tensor(getattr(sample_t, img_type)))
        return sample_t


class Normalize:

    def __init__(self, mean, std, apply_to=['img_rgb']):
        self.mean = mean
        self.std = std
        self.apply_to = apply_to

    def __call__(self, sample):
        sample_t = deepcopy(sample)

        for img_type in self.apply_to:
            print(img_type)
            setattr(sample_t, img_type, F.normalize(getattr(sample_t, img_type), mean=self.mean, std=self.std))
        return sample_t
