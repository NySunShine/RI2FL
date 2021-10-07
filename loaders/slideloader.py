from sys import path
import h5py
import math
import torch
import torch.distributed as dist

import numpy as np
import abc
from pathlib import Path
from itertools import chain
from collections import deque
from scipy.ndimage import zoom
import random
import torch.utils.data as data
from utils.augmentation import RandAugment


class _PatchDatasetBase(data.IterableDataset):
    def __init__(self, path) -> None:
        self.path = path
        self.name = Path(path).stem
        self.zoomed_size = 512
        self.patch_size = 256
        self.cropped_depth = 64
        self.offsets = deque()
        self._load_image()
        self._generate_patch_offset()

    @abc.abstractmethod
    def _generate_patch_offset(self):
        pass

    def _load_image(self):
        with h5py.File(self.path, "r") as h:
            ri = h["ri"][()].astype(np.float32)
            fl = h["fl"][()].astype(np.float32)
        self.images = [ri, fl]

    def _make_a_sample_set(self, beg_y, beg_x):
        aug_fn = RandAugment(1)
        end_y = beg_y + self.patch_size
        end_x = beg_x + self.patch_size

        patches = []
        for image in self.images:
            patches += [
                image[
                    self.z_offset : self.z_offset + self.cropped_depth,
                    beg_y:end_y,
                    beg_x:end_x,
                ]
            ]

        aug_patches = aug_fn(*patches)
        ri, fl = [self._np2tt(patch) for patch in aug_patches]
        return ri, fl

    def __iter__(self):
        for beg_y, beg_x in self.offsets:
            ri, fl = self._make_a_sample_set(beg_y, beg_x)
            yield ri, fl

    @staticmethod
    def _np2tt(x):
        return torch.from_numpy(x).view(1, *x.shape).float()


class RandomPatchDataset(_PatchDatasetBase):
    def __init__(self, path, num_patch=9):
        self.num_patch = num_patch
        super().__init__(path)

    def _generate_patch_offset(self):
        z, y, x = self.images[0].shape
        self.z_offset = random.randint(0, z - self.cropped_depth)

        for _ in range(self.num_patch):
            y_offset = random.randint(0, y - self.patch_size)
            x_offset = random.randint(0, x - self.patch_size)
            self.offsets.append((y_offset, x_offset))


class SlideDataset(_PatchDatasetBase):
    def _generate_patch_offset(self):
        z, y, x = self.images[0].shape
        self.z_offset = (z - self.cropped_depth) // 2
        x_n = y_n = self.patch_size // 2

        for h in range(math.ceil((y - self.patch_size) / y_n) + 1):
            for w in range(math.ceil((x - self.patch_size) / x_n) + 1):
                y_offset = y_n * h
                x_offset = x_n * w
                y_offset = min(max(y_offset, 0), y - self.patch_size)
                x_offset = min(max(x_offset, 0), x - self.patch_size)

                self.offsets.append((y_offset, x_offset))


class SlideInferDataset(SlideDataset):
    def _preprocess(self, x):
        z, _, _ = x.shape

        x = np.pad(
            x,
            (
                (0, 0),
                ((self.patch_size + 1) // 2, self.patch_size // 2),
                ((self.patch_size + 1) // 2, self.patch_size // 2),
            ),
            "symmetric",
        )
        z_offset = (z - self.cropped_depth) // 2
        return x[z_offset : z_offset + self.cropped_depth]

    def _load_image(self):
        with h5py.File(self.path, "r") as h:
            img = h["ri"][()].astype(np.float32)
        self.img = self._preprocess(img)

    def _generate_patch_offset(self):
        _, y, x = self.img.shape
        x_n = y_n = self.patch_size // 2

        for h in range(math.ceil((y - self.patch_size) / y_n) + 1):
            for w in range(math.ceil((x - self.patch_size) / x_n) + 1):
                y_offset = y_n * h
                x_offset = x_n * w
                y_offset = min(max(y_offset, 0), y - self.patch_size)
                x_offset = min(max(x_offset, 0), y - self.patch_size)

                self.offsets.append((y_offset, x_offset))

    def __iter__(self):
        for c_y, c_x in self.offsets:
            img = self.img[:, c_y : c_y + self.patch_size, c_x : c_x + self.patch_size]
            img = torch.from_numpy(img)[None, ...]
            yield img, np.array([c_y, c_x]), self.name


class MpChainDataset(data.ChainDataset):
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id
        for d in self.datasets[worker_id]:
            assert isinstance(
                d, data.IterableDataset
            ), "ChainDataset only supports IterableDataset"
            for x in d:
                yield x


class PatchLoader(object):
    def __init__(self, path, fl_type, batch_size, cpus):
        dataset_list = list(Path(f"{path}/{fl_type}").glob("*.h5"))
        self.trainset = dataset_list[:14]
        self.valset = dataset_list[14:17]
        self.testset = dataset_list[17:]
        self.batch_size = batch_size
        self.cpus = cpus
        self.rank = dist.get_rank()
        self.dist_size = dist.get_world_size()

    def load(self, phase):
        _f = {
            "train": lambda: self._train(),
            "val": lambda: self._val(),
            "test": lambda: self._test(),
        }
        return _f[phase]()

    def _setup_loader(self, dataset_list, dataset_indices, dataset_type):
        dist_size = self.dist_size * self.cpus
        if len(dataset_indices) % dist_size:
            dataset_indices = np.pad(
                dataset_indices,
                (0, dist_size - len(dataset_indices) % dist_size),
                mode="edge",
            )
        self.len = int(len(dataset_indices) * 9 / self.dist_size / self.batch_size)
        cpus = self.cpus
        lists = [dataset_indices[i::cpus] for i in range(cpus)]
        datasets = []
        for l in lists:
            datasets += [
                (
                    dataset_type(dataset_list[i])
                    for idx, i in enumerate(l)
                    if idx % self.dist_size == self.rank
                )
            ]
        dataset = MpChainDataset(datasets)
        loader = data.DataLoader(
            dataset,
            self.batch_size,
            pin_memory=True,
            drop_last=False,
            num_workers=cpus,
        )
        return loader

    def _train(self):
        dataset_indices = np.random.permutation(range(len(self.trainset)))
        return self._setup_loader(self.trainset, dataset_indices, RandomPatchDataset)

    def _val(self):
        dataset_indices = np.array(range(len(self.valset)))
        return self._setup_loader(self.valset, dataset_indices, SlideDataset)

    def _test(self):
        dataset_indices = np.array(range(len(self.testset)))
        return self._setup_loader(self.testset, dataset_indices, SlideDataset)


class SlideInferLoader(object):
    def __init__(self, paths, batch_size, cpus):
        extensions = ["h5", "hdf", "TCF"]
        self.paths = list(chain(*[Path(paths).rglob(f"*.{ext}") for ext in extensions]))
        self.batch_size = batch_size
        self.cpus = cpus
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def load(self):
        lists = [self.paths[i :: self.cpus] for i in range(self.cpus)]
        datasets = []
        for l in lists:
            datasets += [
                (
                    SlideInferDataset(i)
                    for idx, i in enumerate(l)
                    if idx % self.world_size == self.rank
                )
            ]
        dataset = MpChainDataset(datasets)
        loader = data.DataLoader(
            dataset,
            self.batch_size,
            pin_memory=True,
            drop_last=False,
            num_workers=self.cpus,
        )
        return loader
