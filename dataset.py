# RGB-D Salient Object Detection (SOD) dataset
import glob
import os
import random
from typing import List, Callable

from PIL import Image
from torch.utils.data import Dataset


class Sample:
    """The class represents a single sample from the corresponding SOD dataset.

    Args:
        img_rgb: RGB image.
        img_depth: Depth image (grayscale).
        img_gt: Ground truth image (grayscale).
    """

    def __init__(self, img_rgb: Image.Image, img_depth: Image.Image, img_gt: Image.Image):
        self.img_rgb = img_rgb
        self.img_depth = img_depth
        self.img_gt = img_gt


class RgbdSodDataset(Dataset):
    """The class represents a dataset for the Salient Object Detection (SOD) task.

    Args:
        paths_to_datasets: List of paths to datasets.
        transform_fn: Transformation function applied to a single sample consisting of an RGB image, depth image and
                      ground truth image.
        max_samples: Maximum number of samples in the dataset. All of the available data will be used if set to 0.
        in_memory: The dataset will be stored in RAM if set to True, otherwise the samples will be streamed from
                   a disk.
    """

    def __init__(self, paths_to_datasets: List[str], transform_fn: Callable = None, max_samples: int = 0,
                 in_memory: bool = False):
        self.list_fnames_rgb = []
        self.list_fnames_gt = []
        self.list_fnames_depth = []

        for dataset in paths_to_datasets:
            ids = sorted(glob.glob(os.path.join(dataset, 'RGB', '*.jpg')))
            ids = [os.path.splitext(os.path.basename(sample_id))[0] for sample_id in ids]
            for sample_id in ids:
                self.list_fnames_rgb.append(os.path.join(dataset, 'RGB', sample_id + '.jpg'))
                self.list_fnames_gt.append(os.path.join(dataset, 'GT', sample_id + '.png'))
                self.list_fnames_depth.append(os.path.join(dataset, 'depth', sample_id + '.png'))

        if 0 < max_samples < len(self.list_fnames_rgb):
            indices = random.sample(range(len(self.list_fnames_rgb)), max_samples)
            self.list_fnames_rgb = [self.list_fnames_rgb[i] for i in indices]
            self.list_fnames_gt = [self.list_fnames_gt[i] for i in indices]
            self.list_fnames_depth = [self.list_fnames_depth[i] for i in indices]

        self.transform_fn = transform_fn
        self.in_memory = in_memory

        if in_memory:
            self.samples = []
            for index in range(len(self.list_fnames_rgb)):
                self.samples.append(self._get_sample(index))

    def _get_sample(self, index: int) -> Sample:
        # Loads a single sample.
        img_rgb = Image.open(self.list_fnames_rgb[index]).convert('RGB')
        img_gt = Image.open(self.list_fnames_gt[index]).convert('L')
        img_depth = Image.open(self.list_fnames_depth[index]).convert('L')
        return Sample(img_rgb=img_rgb, img_depth=img_depth, img_gt=img_gt)

    def __len__(self) -> int:
        return len(self.list_fnames_rgb)

    def __getitem__(self, index: int) -> Sample:
        if self.in_memory:
            sample = self.samples[index]
        else:
            sample = self._get_sample(index)
        if self.transform_fn:
            sample = self.transform_fn(sample)
        return sample
