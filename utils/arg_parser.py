import os
import yaml

import torch
import torch.distributed as dist
from apex.parallel import DistributedDataParallel


torch.backends.cudnn.benchmark = True


class Argments(object):
    @staticmethod
    def _file_load(yaml_file):
        with open(fr'{yaml_file}') as f:
            y = yaml.safe_load(f)
        return y

    def _model_load(self):
        fl_type = self['setup/fl_type']
        # model = torch.nn.DataParallel(torch.jit.load(f"{self['path/model_path']}/{fl_type}.pth")).cuda()
        model = DistributedDataParallel(torch.jit.load(f"{self['path/model_path']}/{fl_type}.pth").cuda())
        self['model'] = model

    def __init__(self, yaml_file, cmd_args):
        self.file_name = yaml_file
        self._y = self._file_load(yaml_file)
        os.environ["CUDA_VISIBLE_DEVICES"] = self["setup/gpus"]
        self['setup/rank'] = cmd_args.local_rank

        torch.cuda.set_device(cmd_args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')

    def reset(self):
        for k, v in list(self.__dict__.items()):
            if 'model' in k:
                del self.__dict__[k]
        torch.cuda.empty_cache()
        self._model_load()

    def _get(self, *keys):
        v = self._y
        for k in keys:
            v = v[k]
        return v

    def _update(self, *keys, value):
        k = self._y
        for i in range(len(keys) - 1):
            k.setdefault(keys[i], {})
            k = k[keys[i]]
        k[keys[-1]] = value

    def __str__(self):
        return f'{self.file_name}\n{self._y}'

    def __contains__(self, item):
        def search_recursively(d, t):
            for k, v in d.items():
                if k == t:
                    return True
                elif isinstance(v, dict):
                    search_recursively(v, t)
            return False

        return search_recursively(self._y, item)

    def __getitem__(self, key):
        return self._get(*key.split('/'))

    def __setitem__(self, key, value):
        self._update(*key.split('/'), value=value)
