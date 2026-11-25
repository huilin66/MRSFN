# -*- coding: utf-8 -*-
"""Dataset for the 128 x 128 WHU-H2SR-MT patches.

The H2SR patch lists produced by ``tools/h2sr_tools.py`` have four columns::

    hsi/<id>.tif  mt/<id>.tif  lbl/<id>.tif  mt_lbl/<id>.tif

Static segmentation baselines use a six-channel MSI composite (the temporal
median of the 31 Sentinel-2 dates, bands B4/B3/B2/B8/B11/B12) stored in
``msi/<id>.tif``.  This class therefore reads the HSI and label columns and
resolves the MSI path from the sample id, so the existing ``train.txt`` and
``val.txt`` lists do not need to be rewritten.
"""

from __future__ import print_function
from __future__ import division

import os
from pathlib import Path

import paddle
import numpy as np
from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose


@manager.DATASETS.add_component
class RS_H2SR(paddle.io.Dataset):
    """Dual-input dataset: [6-channel MSI composite, 249-band HSI, label]."""

    def __init__(self,
                 transforms,
                 dataset_root,
                 num_classes=8,
                 mode='train',
                 train_path=None,
                 val_path=None,
                 test_path=None,
                 separator=' ',
                 ignore_index=255,
                 msi_dir='msi',
                 edge=False):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        self.file_list = list()
        self.mode = mode.lower()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.msi_dir = msi_dir
        self.edge = edge

        if self.mode not in ['train', 'val', 'test']:
            raise ValueError(
                "mode should be 'train', 'val' or 'test', but got {}.".format(mode))
        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")
        if not os.path.exists(self.dataset_root):
            raise FileNotFoundError('there is not `dataset_root`: {}.'.format(
                self.dataset_root))

        if mode == 'train':
            file_path = train_path
        elif mode == 'val':
            file_path = val_path
        else:
            file_path = test_path
        if file_path is None:
            raise ValueError('`{}_path` is necessary.'.format(mode))
        if not os.path.exists(file_path):
            raise FileNotFoundError('`{}_path` is not found: {}'.format(mode, file_path))

        msi_root = Path(self.dataset_root) / self.msi_dir
        with open(file_path, 'r') as f:
            for line in f:
                items = line.strip().split(separator)
                if len(items) != 4:
                    raise ValueError(
                        "RS_H2SR expects 4 columns "
                        "(hsi mt lbl mt_lbl), got: {!r}".format(line.strip()))
                hsi_rel, _mt_rel, lbl_rel, _mt_lbl_rel = items
                stem = Path(hsi_rel).stem
                im1_path = str(msi_root / (stem + '.tif'))
                im2_path = os.path.join(self.dataset_root, hsi_rel)
                label_path = os.path.join(self.dataset_root, lbl_rel)
                if not os.path.exists(im1_path):
                    raise FileNotFoundError(
                        "MSI composite not found: {}. Run "
                        "`python tools/h2sr_tools.py build-msi` first.".format(im1_path))
                self.file_list.append([im1_path, im2_path, label_path])

    def __getitem__(self, idx):
        image1_path, image2_path, label_path = self.file_list[idx]
        if self.mode == 'test':
            im1, im2, _ = self.transforms(
                im1_path=image1_path, im2_path=image2_path)
            im1 = im1[np.newaxis, ...]
            im2 = im2[np.newaxis, ...]
            return im1, im2, image1_path, image2_path
        elif self.mode == 'val':
            im1, im2, label = self.transforms(
                im1_path=image1_path, im2_path=image2_path, label_path=label_path)
            return im1, im2, label
        else:
            im1, im2, label = self.transforms(
                im1_path=image1_path, im2_path=image2_path, label_path=label_path)
            if self.edge:
                from paddleseg.transforms import functional as F
                edge_mask = F.mask_to_binary_edge(
                    label, radius=2, num_classes=self.num_classes)
                return im1, im2, label, edge_mask
            return im1, im2, label

    def __len__(self):
        return len(self.file_list)
