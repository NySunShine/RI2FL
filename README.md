# RI2FL
This repository contains Pytorch implementation of RI2FL.

## Usage
### YAML Script
RI2FL class parses arguments from the `yaml_file`. 

Exmaple of `yaml_file`:

```yaml
# scripts/infer.yaml
path:
    dataset: testset
    model_path: models
    save_path: result

setup:
    fl_list: mem, nuc, oli
    batch_size: 4
    gpus: 0,1,2,3
    cpus: 4
    zoomed_size: [512, 512]
    patch_size: 256
    cropped_depth: 64
    num_drop: 0
    num_tta: 4
```
### Inference
```python
# example.py
import argparse
from ri2fl import Ri2Fl
import torch.distributed as dist


argparser = argparse.ArgumentParser()
argparser.add_argument("yaml")
argparser.add_argument("--local_rank", default=0, type=int)
cmd_args = argparser.parse_args()
ri2fl = Ri2Fl(f"{cmd_args.yaml}.yaml", cmd_args)
ri2fl.predict_all()
dist.destroy_process_group()
```

Then, run the python script with the following command as bellow.
```bash
➜ python -m torch.distributed.launch --nporoc_per_node=4 example.py infer
```