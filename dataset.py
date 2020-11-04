import glob
import os
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class RgbdSodDataset(Dataset):

    def __init__(self, datasets, transform_fn=None, sample_limit=0, in_memory=False):
        if not isinstance(datasets, list):
            datasets = [datasets]
        self.list_images = []
        self.list_gts = []
        self.list_depths = []

        for dataset in datasets:
            ids = sorted(glob.glob(os.path.join(dataset, 'RGB', '*.jpg')))
            ids = [os.path.splitext(os.path.split(sample_id)[1])[0] for sample_id in ids]
            for sample_id in ids:
                self.list_images.append(os.path.join(dataset, 'RGB', sample_id + '.jpg'))
                self.list_gts.append(os.path.join(dataset, 'GT', sample_id + '.png'))
                self.list_depths.append(os.path.join(dataset, 'depth', sample_id + '.png'))

        if sample_limit != 0 and len(self.list_images) > sample_limit:
            indices = random.sample(range(len(self.list_images)), sample_limit)
            self.list_images = [self.list_images[i] for i in indices]
            self.list_gts = [self.list_gts[i] for i in indices]
            self.list_depths = [self.list_depths[i] for i in indices]

        self.transform_fn = transform_fn
        self.in_memory = in_memory

        if in_memory:
            self.samples = []

            for index in range(len(self.list_images)):
                self.samples.append(self._get_sample(index))

    def _get_sample(self, index):
        img = np.array(Image.open(self.list_images[index]).convert('RGB'))
        gt = np.array(Image.open(self.list_gts[index]).convert('L'))
        depth = np.array(Image.open(self.list_depths[index]).convert('L'))

        sample = {'img': img, 'gt': gt, 'depth': depth,
                  'metadata': {'id': os.path.splitext(os.path.split(self.list_gts[index])[1])[0]}}
        sample['metadata']['source_size'] = np.array(gt.shape[::-1])
        sample['metadata']['img_path'] = self.list_images[index]
        sample['metadata']['gt_path'] = self.list_gts[index]
        sample['metadata']['depth_path'] = self.list_depths[index]
        return sample

    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, index):
        if self.in_memory:
            sample = self.samples[index]
        else:
            sample = self._get_sample(index)
        if self.transform_fn:
            sample = self.transform_fn(sample)
        return sample
