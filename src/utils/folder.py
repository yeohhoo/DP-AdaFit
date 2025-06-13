#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from logging import getLogger
import torch.utils.data as data

from PIL import Image

import os
import sys
import time
import pickle
from os.path import basename, dirname, join
import torch
import numpy as np

logger = getLogger()

CACHE_DATASET = {
    '/datasets01/imagenet_full_size/061417/train':
    '/checkpoint/asablayrolles/imagenet_train.pkl',

    '/datasets01/imagenet_full_size/061417/val':
    '/checkpoint/asablayrolles/imagenet_val.pkl',

    '/checkpoints/tomsander/imagenet/imagenet_full_size/061417/train':
    '/checkpoints/tomsander/imagenet_train2.pkl',

    "H:\\dataset\\imagenet\\imagenet\\train":
    "H:\\dataset\\imagenet\\imagenet\\train\\imagenet_train.pkl",
    
    "H:\\dataset\\imagenet\\imagenet\\val":
    "H:\\dataset\\imagenet\\imagenet\\val\\imagenet_val.pkl",

    # '/checkpoints/tomsander/imagenet/imagenet_full_size/061417/val':
    # '/checkpoints/tomsander/imagenet_val2.pkl',

    "E:\\PyProjects\\year_2024\\cifar_train":
    "E:\\PyProjects\\year_2024\\cifar_train\\cifar10_train.pkl",

    "E:\\PyProjects\\year_2024\\cifar_test":
    "E:\\PyProjects\\year_2024\\cifar_test\\cifar10_test.pkl",

    "/hy-tmp/imagenet/train":
    "/hy-tmp/imagenet/train/imagenet_train.pkl",

    "/hy-tmp/imagenet/val":
    "/hy-tmp/imagenet/val/imagenet_val.pkl",

    "/hy-tmp/cifar_train":
    "/hy-tmp/cifar_train/cifar10_train.pkl",

    "/hy-tmp/cifar_test":
    "/hy-tmp/cifar_test/cifar10_test.pkl",

    "/users/u2020010337/guided-diffusion/cifar10_l_train":
    "/users/u2020010337/guided-diffusion/cifar10_l_train/cifar10_train.pkl",

    "/users/u2020010337/guided-diffusion/cifar10_l_test":
    "/users/u2020010337/guided-diffusion/cifar10_l_test/cifar10_test.pkl",

}


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp',
                  '.pgm', '.tif', '.tiff', 'webp']


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self,
                 root,
                 loader=default_loader,
                 extensions=IMG_EXTENSIONS,
                 transform=None,
                 target_transform=None):

        start = time.time()
        path_cache = CACHE_DATASET[root]
        if not os.path.isfile(path_cache):
            print("Images cache not found in %s, \
                  parsing dataset..." % path_cache,
                  file=sys.stderr)
            classes, class_to_idx = self._find_classes(root)
            samples = make_dataset(root, class_to_idx, extensions)
            print("Parsing image folder took %.2f \
                  seconds." % (time.time() - start),
                  file=sys.stderr)
            with open(path_cache, "wb") as f:
                pickle.dump((classes, class_to_idx, samples), f)
        else:
            with open(path_cache, "rb") as f:
                classes, class_to_idx, samples = pickle.load(f)
            print("Loading cached images took %.2f \
                  seconds." % (time.time() - start),
                  file=sys.stderr)

        print("Dataset contains %i images." % len(samples), file=sys.stderr)

        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: \
                                " + root + "\n"
                                "Supported extensions are: "
                                + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx

        self.samples = samples
        self.targets = np.array([s[1] for s in samples])

        self.transform = transform
        self.target_transform = target_transform

        # Samples are grouped by class contiguously
        assert np.all(0 <= self.targets[1:] - self.targets[:-1])
        assert np.all(self.targets[1:] - self.targets[:-1] <= 1)
        assert np.sum(self.targets[1:]
                      - self.targets[:-1]) == max(self.targets)

        cl_positions = np.nonzero(self.targets[1:]
                                  - self.targets[:-1])[0] + 1
        cl_positions = np.insert(cl_positions, 0, 0)
        cl_positions = np.append(cl_positions, len(self.targets))

        self.class2position = {i: np.arange(cl_positions[i], cl_positions[i+1])
                               for i in range(len(cl_positions) - 1)}
        assert all([all(self.targets[v] == i
                    for v in self.class2position[i])
                    for i in range(max(self.targets) + 1)])
        
        self.image_paths, self.labels = zip(*self.samples)
        self.image_paths = np.array(list(self.image_paths))
        self.labels = np.array(list(self.labels))
        # assert zip(self.image_paths.tolist(), self.labels.tolist()) == self.samples

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir),
                   and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir)
                       if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index
                   of the target class.
        """
        path = self.image_paths[index] # to avoid copy-on-access
        target = self.labels[index] # to avoid copy-on-access
        # path, target = self.samples[index]

        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # use for guided-diffusion model
        out_dict = {}
        # if self.local_classes is not None:
        out_dict["y"] = np.array(target, dtype=np.int64)

        # return sample, target
        return sample, out_dict

    def __len__(self):
        return len(self.image_paths) # to avoid copy-on-access
        # return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp,
            self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp))
        )
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp,
            self.target_transform.__repr__().replace('\n',
                                                     '\n' + ' ' * len(tmp))
        )
        return fmt_str


class MultiViewDatasetFolder(DatasetFolder):
    def __init__(self,
                 root,
                 loader=default_loader,
                 extensions=IMG_EXTENSIONS,
                 transform=None,
                 target_transform=None):
        super().__init__(root,
                         loader,
                         extensions,
                         transform=transform,
                         target_transform=target_transform)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index
                   of the target class.
        """
        # path, target = self.samples[index] 
        path = self.image_paths[index] # to avoid copy-on-access
        target = self.labels[index] # to avoid copy-on-access

        sample = self.loader(path)
        if self.transform is not None:
            samples = [t(sample) for t in self.transform]
        if self.target_transform is not None:
            target = self.target_transform(target)
        samples = torch.stack(samples)

        # use for guided-diffusion model
        out_dict = {}
        # if self.local_classes is not None:
        out_dict["y"] = np.array(target, dtype=np.int64)

        return samples, out_dict

        # return samples, target

    def __len__(self):
        return len(self.image_paths) # to avoid copy-on-access 
        # return len(self.samples)
