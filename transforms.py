# Image transformation functions
import abc
from copy import deepcopy
from functools import partial
from typing import List, Callable, Tuple, Union

import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image

from dataset import Sample


class BaseTransform(abc.ABC):
    """Abstract base class for image transformation functions.

    Args:
        apply_to: List of image types to be transformed. Possible values are: 'img_rgb', 'img_depth' and 'img_gt'.
        prob: Probability of applying the transformation function.

    """

    def __init__(self, apply_to: List[str], prob: float):
        self.apply_to = apply_to
        self.prob = prob

    def _apply(self, sample: Sample, func: Callable) -> Sample:
        # Applies the provided transformation function to the specified image types in the sample.
        # Determines if the transformation is applicable with respect to the provided probability.
        if np.random.random(1) < self.prob:
            sample_t = deepcopy(sample)
            for img_type in self.apply_to:
                setattr(sample_t, img_type, func(getattr(sample_t, img_type)))
            return sample_t
        else:
            return sample

    @abc.abstractmethod
    def __call__(self, sample: Sample) -> Sample:
        pass


class Resize(BaseTransform):
    """Resizes the image. The transformation should be applied to all image types:
    'img_rgb', 'img_depth', and 'img_gt' hence the default value of 'apply_to' parameter is set to all image types.

    Args:
        apply_to: List of image types to be transformed. Possible values are: 'img_rgb', 'img_depth' and 'img_gt'.
        prob: Probability of applying the transformation function.

    """

    def __init__(self, size: Union[Tuple[int, int], int], apply_to: List[str] = ['img_rgb', 'img_depth', 'img_gt'],
                 prob: float = 1.0):
        super().__init__(apply_to, prob)
        self.size = size
        self.prob = prob

    def __call__(self, sample: Sample) -> Sample:
        func = T.Resize(self.size)
        return self._apply(sample, func)


class RandomCrop(BaseTransform):
    """Crops the image at random location. The transformation should be applied to all image types:
    'img_rgb', 'img_depth', and 'img_gt' hence the default value of 'apply_to' parameter is set to all image types.

    Args:
        apply_to: List of image types to be transformed. Possible values are: 'img_rgb', 'img_depth' and 'img_gt'.
        prob: Probability of applying the transformation function.

    """

    def __init__(self, size: Union[Tuple[int, int], int], apply_to: List[str] = ['img_rgb', 'img_depth', 'img_gt'],
                 prob: float = 1.0):
        super().__init__(apply_to, prob)
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        self.prob = prob

    def __call__(self, sample: Sample) -> Sample:
        w, h = sample.img_rgb.size
        h_crop, w_crop = self.size
        coord_i = np.random.randint(0, h - h_crop + 1)
        coord_j = np.random.randint(0, w - w_crop + 1)
        func = partial(F.crop, top=coord_i, left=coord_j, height=h_crop, width=w_crop)
        return self._apply(sample, func)


class RandomHorizontalFlip(BaseTransform):
    """Horizontally flips the image. The transformation should be applied to all image types:
    'img_rgb', 'img_depth', and 'img_gt' hence the default value of 'apply_to' parameter is set to all image types.

    Args:
        apply_to: List of image types to be transformed. Possible values are: 'img_rgb', 'img_depth' and 'img_gt'.
        prob: Probability of applying the transformation function.

    """

    def __init__(self, apply_to: List[str] = ['img_rgb', 'img_depth', 'img_gt'], prob: float = 0.5):
        super().__init__(apply_to, prob)
        self.prob = prob

    def __call__(self, sample: Sample) -> Sample:
        return self._apply(sample, F.hflip)


class RandomRotation(BaseTransform):
    """Rotates the image by an angle which is randomly selected from the provided range (via 'degrees' parameter).
    The transformation should be applied to all image types: 'img_rgb', 'img_depth', and 'img_gt' hence the
    default value of 'apply_to' parameter is set to all image types.

    Args:
        degrees: Range of degrees to select from. If degrees is a number instead of sequence like (min, max),
                 the range of degrees will be (-degrees, +degrees).
        apply_to: List of image types to be transformed. Possible values are: 'img_rgb', 'img_depth' and 'img_gt'.
        prob: Probability of applying the transformation function.

    """

    def __init__(self, degrees: Union[Tuple[int, int], int], apply_to: List[str] = ['img_rgb', 'img_depth', 'img_gt'],
                 prob: float = 0.5):
        super().__init__(apply_to, prob)
        if isinstance(degrees, int):
            degrees = (-degrees, degrees)
        self.degrees = degrees
        self.prob = prob

    def __call__(self, sample: Sample) -> Sample:
        angle = np.random.randint(self.degrees[0], self.degrees[1])
        func = partial(F.rotate, angle=angle)
        return self._apply(sample, func)


class GaussianNoise(BaseTransform):
    """Adds a Gaussian noise to selected image types from a single sample.

    Args:
        intensity: Intensity of the noise.
        apply_to: List of image types to be transformed. Possible values are: 'img_rgb', 'img_depth' and 'img_gt'.
        prob: Probability of applying the transformation function.

    """

    def __init__(self, intensity: float, apply_to: List[str], prob: float = 0.5):
        super().__init__(apply_to, prob)
        self.intensity = intensity

    def _add_noise(self, img: Image.Image) -> Image.Image:
        arr_img = np.array(img)
        arr_noise = np.random.normal(0, 2.5, arr_img.shape)
        arr_noisy_img = np.clip((arr_img + self.intensity * arr_noise), 0, 255).astype(np.uint8)
        return Image.fromarray(arr_noisy_img)

    def __call__(self, sample: Sample) -> Sample:
        return self._apply(sample, self._add_noise)


class ToTensor(BaseTransform):
    """Converts PIL images from a single sample to Pytorch tensors.

    Args:
        apply_to: List of image types to be transformed. Possible values are: 'img_rgb', 'img_depth' and 'img_gt'.

    """

    def __init__(self, apply_to: List[str] = ['img_rgb', 'img_depth', 'img_gt']):
        super().__init__(apply_to, 1.0)

    def __call__(self, sample: Sample) -> Sample:
        func = F.to_tensor
        return self._apply(sample, func)


class Normalize(BaseTransform):
    """Normalizes tensors from a single sample with mean and standard deviation.

    Args:
        mean: List of means for each channel.
        std: List of standard deviations for each channel.
        apply_to: List of image types to be transformed. Possible values are: 'img_rgb', 'img_depth' and 'img_gt'.

    """

    def __init__(self, mean: List[float], std: List[float], apply_to: List[str] = ['img_rgb']):
        super().__init__(apply_to, 1.0)
        self.mean = mean
        self.std = std

    def __call__(self, sample: Sample) -> Sample:
        func = partial(F.normalize, mean=self.mean, std=self.std)
        return self._apply(sample, func)
