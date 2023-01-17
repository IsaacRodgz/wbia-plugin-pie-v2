# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

# import os
import torch
from datasets.sampler import RandomCopiesIdentitySampler
from datasets import init_image_dataset
from .transforms import build_train_test_transforms

# from utils.tools import write_json


class AnimalImageDataManager:
    r"""Image data manager for animal data.

    Args:
        root (str): root path to datasets.
        source (str): source dataset.
        height (int, optional): target image height. Default is 256.
        width (int, optional): target image width. Default is 128.
        transforms (str or list of str, optional): transformations applied to
                    model training. Default is 'random_flip'.
        k_tfm (int): number of times to apply augmentation to an image
            independently. If k_tfm > 1, the transform function will be
            applied k_tfm times to an image. This variable will only be
            useful for training and is currently valid for image datasets only.
        norm_mean (list or None, optional): data mean.
                    Default is None (use imagenet mean).
        norm_std (list or None, optional): data std.
                    Default is None (use imagenet std).
        use_gpu (bool, optional): use gpu. Default is True.
        batch_size_train (int, optional): number of images in a training batch.
                    Default is 32.
        batch_size_test (int, optional): number of images in a test batch.
                    Default is 32.
        workers (int, optional): number of workers. Default is 4.
        num_instances (int, optional): number of instances per identity in a
                    batch. Default is 4.
        num_copies (int, optional): number of copies for each image (different
                    augmentation will be applied). Default is 1.
        config_fpath (str): path to config file for config-defined datasets

    """
    data_type = 'image'

    def __init__(
        self,
        root='',
        source=None,
        height=256,
        width=128,
        transforms_train='random_flip',
        transforms_test='resize',
        k_tfm=1,
        norm_mean=None,
        norm_std=None,
        use_gpu=True,
        batch_size_train=32,
        batch_size_test=32,
        workers=2,
        num_instances=4,
        num_copies=1,
        config_fpath='',
    ):

        self.source = source
        self.height = height
        self.width = width

        self.transform_tr, self.transform_te = build_train_test_transforms(
            self.height,
            self.width,
            transforms_train=transforms_train,
            transforms_test=transforms_test,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.use_gpu = torch.cuda.is_available() and use_gpu

        self.num_copies = num_copies

        print('=> Loading train dataset')
        self.train_set = init_image_dataset(
            self.source, transform=self.transform_tr, k_tfm=k_tfm, mode='train', root=root, config_fpath=config_fpath
        )

        self._num_train_pids = self.train_set.num_train_pids

        self.train_loader = torch.utils.data.DataLoader(
            self.train_set,
            sampler=RandomCopiesIdentitySampler(
                self.train_set.train,
                batch_size=batch_size_train,
                num_instances=num_instances,
                num_copies=num_copies,
            ),
            batch_size=batch_size_train,
            shuffle=False,
            num_workers=workers,
            pin_memory=self.use_gpu,
            drop_last=True,
        )

        print('=> Loading test dataset')

        self.test_set = init_image_dataset(
            self.source, transform=self.transform_te, mode='test', root=root, config_fpath=config_fpath
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_set,
            batch_size=batch_size_test,
            shuffle=False,
            num_workers=workers,
            pin_memory=self.use_gpu,
            drop_last=False,
        )

    @property
    def num_train_pids(self):
        """Returns the number of training person identities."""
        return self._num_train_pids

    def preprocess_pil_img(self, img):
        """Transforms a PIL image to torch tensor for testing."""
        return self.transform_te(img)
