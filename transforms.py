# Transformation functions
import abc
from copy import deepcopy
from functools import partial

import torchvision.transforms.functional as F


class BaseTransform(abc.ABC):

    def __init__(self, apply_to):
        self.apply_to = apply_to

    def _apply(self, sample, func):
        sample_t = deepcopy(sample)

        for img_type in self.apply_to:
            setattr(sample_t, img_type, func(getattr(sample_t, img_type)))
        return sample_t

    @abc.abstractmethod
    def __call__(self, sample):
        pass


class ToTensor(BaseTransform):

    def __init__(self, apply_to=['img_rgb', 'img_depth', 'img_gt']):
        super().__init__(apply_to)

    def __call__(self, sample):
        func = F.to_tensor
        return self._apply(sample, func)


class Normalize(BaseTransform):

    def __init__(self, mean, std, apply_to=['img_rgb']):
        super().__init__(apply_to)
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        func = partial(F.normalize, mean=self.mean, std=self.std)
        return self._apply(sample, func)
