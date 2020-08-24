import h5py
import math
import torch
import torch.distributed as dist

import numpy as np
from pathlib import Path
from itertools import chain
from collections import deque
from scipy.ndimage import zoom

import torch.utils.data as data


class SlideDataset(data.IterableDataset):
    def __init__(self, path, zoomed_size, patch_size, cropped_depth):
        self.path = path
        self.name = Path(path).stem
        self.zoomed_size = zoomed_size
        self.patch_size = patch_size
        self.cropped_depth = cropped_depth
        self.offsets = deque()
        self._load_image()
        self._generate_patch_offset()

    def _preprocess(self, x):
        def norm(_x):
            if np.all([_x >= 0., _x <= 1.]):
                return _x
            if _x.min() > 10000:
                return (_x.clip(13370, 13900) - 13370) / (13900 - 13370)
            return (_x.clip(1.337, 1.39) - 1.337) / (1.39 - 1.337)

        x = norm(x)
        d, h, w = x.shape
        _h, _w = self.zoomed_size
        if self.zoomed_size != x.shape[1:]:
            x = zoom(x, (1, _h / h, _w / w), order=1)
        z_pad = 0
        if d < self.cropped_depth:
            z_pad = self.cropped_depth - d

        x = np.pad(x,
                   (((z_pad + 1) // 2, z_pad // 2),
                    ((self.patch_size + 1) // 2, self.patch_size // 2),
                    ((self.patch_size + 1) // 2, self.patch_size // 2)),
                   'symmetric')
        center = d // 2
        return x[center - self.cropped_depth // 2: center + self.cropped_depth // 2]

    def _load_image(self):
        with h5py.File(self.path, 'r') as h:
            img = h['ri'][()].astype(np.float32)
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
            img = self.img[:,
                           c_y: c_y + self.patch_size,
                           c_x: c_x + self.patch_size]
            img = torch.from_numpy(img)[None, ...]
            yield img, np.array([c_y, c_x]), self.name


class MpChainDataset(data.ChainDataset):
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id
        for d in self.datasets[worker_id]:
            assert isinstance(d, data.IterableDataset), \
                "ChainDataset only supports IterableDataset"
            for x in d:
                yield x


class SlideInferLoader(object):
    def __init__(self, paths, zoomed_size, patch_size, cropped_depth, batch_size, cpus):
        extensions = ['h5', 'hdf', 'TCF']
        self.paths = list(chain(*[Path(paths).rglob(f'*.{ext}') for ext in extensions]))
        self.zoomed_size = zoomed_size
        self.patch_size = patch_size
        self.cropped_depth = cropped_depth
        self.batch_size = batch_size
        self.cpus = cpus
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def load(self):
        lists = [self.paths[i::self.cpus] for i in range(self.cpus)]
        datasets = []
        for l in lists:
            datasets += [(SlideDataset(
                i, self.zoomed_size, self.patch_size, self.cropped_depth
            ) for idx, i in enumerate(l) if idx % self.world_size == self.rank)]
        dataset = MpChainDataset(datasets)
        loader = data.DataLoader(dataset, self.batch_size, pin_memory=True,
                                 drop_last=False, num_workers=self.cpus)
        return loader
