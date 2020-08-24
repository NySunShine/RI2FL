import h5py
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict


def transform(num):
    case = {0: [lambda x: x, lambda x: x],
            1: [lambda x: x.transpose(3, 4).flip(3), lambda x: x.transpose(3, 4).flip(4)],
            2: [lambda x: x.flip(3).flip(4), lambda x: x.flip(3).flip(4)],
            3: [lambda x: x.transpose(3, 4).flip(4), lambda x: x.transpose(3, 4).flip(3)]}

    return case[num]


def noise(num):
    case = {0: lambda x: x,
            1: lambda x: x ** 1.2,
            2: lambda x: x ** 0.8,
            3: lambda x: x + torch.randn_like(x) * 0.015,
            4: lambda x: (x * 1000).to(torch.int).to(torch.float) / 1000}

    return case[num]


def apply_dropout(m):
    if type(m) == torch.nn.Dropout:
        m.train()
        m.p = 0.3


class Predictor(object):
    """
    Runner class for inference of RI2FL.


    Args :
        model (torch.nn.Module) : RI2FL model with trained parameters
        loader (SlideInferLoader) : loader with padding, patching and batching
        stitcher (Stitcer) : stitcher for making a whole volume using a number of patch volumes
        num_drop (int) : a iteration size of MC-Dropout
        num_tta (int) : a iteration size of Test Time Augmentation
        save_path (str) : parent path for saving stitched volume
        fl_type (str) : one of ['mem', 'act', 'mito', 'lipid', 'nuc', 'oli']

    Example::
        >>> runner = Runner(arg['model'], loader, stitcher, setup['num_drop'], setup['num_tta'], save_path, fl_type)
        >>> runner.infer()
    """
    def __init__(self, model, loader, stitcher, num_drop, num_tta, save_path, fl_type):
        self.model = model
        self.loader = loader
        self.stitcher = stitcher
        self.num_drop = num_drop
        self.num_tta = num_tta
        assert num_drop or num_tta, "At least one of [num_drop, num_tta] should be larger than zero."
        self.ns = np.random.randint(0, 5, num_tta)
        self.ts = np.random.randint(0, 4, num_tta)
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.fl_type = fl_type
        self.patches = defaultdict(dict)
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()

    def _tta(self, x, n, t):
        """
        Calculate MC integration result (predictive mean volume) using TTA.
        """
        self.model.eval()
        xn = noise(n)(x).clamp(0, 1).contiguous()
        xn_t = transform(t)[0](xn)
        y_t = self.model(xn_t)

        y = transform(t)[1](y_t)
        return y

    @torch.no_grad()
    def infer_patch(self):
        for img, coord, name in self.loader.load():
            img = img.cuda()
            sum_alea = 0
            sum_epis = 0
            mean_alea = 0
            mean_epis = 0

            if self.num_tta:
                for n, t in zip(self.ns, self.ts):
                    y = self._tta(img, n, t)
                    sum_alea += y
                mean_alea = sum_alea / self.num_tta

            if self.num_drop:
                self.model.apply(apply_dropout)
                for _ in range(self.num_drop):
                    y = self.model(img)
                    sum_epis += y
                mean_epis = sum_epis / self.num_drop

            output = (mean_alea + mean_epis) / 2

            for i in range(output.size(0)):
                f = name[i]
                c = coord[i].numpy()
                o = output[i].cpu().numpy()[0]

                patches = {f'fl_{self.fl_type}': o}

                def save_h5(name, coord, patch):
                    with h5py.File(name, 'a') as h:
                        dst = h.create_group(f"{coord[0]}_{coord[1]}")
                        for k, v in patch.items():
                            dst.create_dataset(k, data=v)
                try:
                    save_h5(f'{self.save_path}/{f}.h5', c, patches)
                except ValueError:
                    pass

    def stitch(self):
        paths = list(Path(self.save_path).glob('*.h5'))[self.rank::self.world_size]
        for p in paths:
            print(p.stem)
            patches = {}
            with h5py.File(p, 'a') as h:
                for c, imgs in h.items():
                    if isinstance(imgs, h5py.Group):
                        y, x = c.split('_')
                        c_ = torch.tensor([int(y), int(x)], dtype=torch.int16)
                        patches[c_] = imgs
                img = self.stitcher.stitch(patches)

                for k, v in img.items():
                    try:
                        data = (v - v.min()) / (v.max() - v.min())
                        h.create_dataset(k, data=data)
                    except RuntimeError:
                        pass

                for k, v in h.items():
                    if isinstance(v, h5py.Group):
                        del h[k]

    @torch.no_grad()
    def infer(self):
        for img, coord, name in self.loader.load():
            img = img.cuda()
            sum_alea = 0
            # sum_alea_sq = 0
            sum_epis = 0
            # sum_epis_sq = 0

            if self.num_tta:
                for n, t in zip(self.ns, self.ts):
                    y = self._tta(img, n, t)
                    sum_alea += y
                    # sum_alea_sq += y ** 2
                mean_alea = sum_alea / self.num_tta
                # mean_alea_sq = sum_alea_sq / self.num_tta
                # alea_std = (mean_alea_sq - mean_alea ** 2)
            else:
                mean_alea = 0

            if self.num_drop:
                self.model.apply(apply_dropout)
                for _ in range(self.num_drop):
                    y = self.model(img)
                    sum_epis += y
                    # sum_epis_sq += y ** 2
                mean_epis = sum_epis / self.num_drop
                # mean_epis_sq = sum_epis_sq / self.num_drop
                # epis_std = (mean_epis_sq - mean_epis ** 2)
            else:
                mean_epis = 0

            output = (mean_alea + mean_epis) / 2

            for i in range(output.size(0)):
                f = name[i]
                c = coord[i]
                o = output[i].cpu().numpy()[0]
                # a = alea_std[i].cpu().numpy()[0]
                # e = epis_std[i].cpu().numpy()[0]

                imgs = {f'fl_{self.fl_type}': o,
                        # f'{self.fl_type}_aleatoric': a,
                        # f'{self.fl_type}_epistemic': e
                        }
                self.patches[f][c] = imgs

                if len(self.patches[f]) == self.stitcher.full_index:
                    def stitch_and_save(file_name, patches):
                        img = self.stitcher.stitch(patches)
                        print(file_name)
                        with h5py.File(file_name, 'a') as h:
                            for k, v in img.items():
                                data = (v - v.min()) / (v.max() - v.min())
                                try:
                                    del h[k]
                                except KeyError:
                                    pass
                                h.create_dataset(k, data=data)

                    stitch_and_save(f"{self.save_path}/{f}.h5", self.patches[f])
