import torch.distributed as dist
import re
import argparse
from loaders.slideloader import SlideInferLoader
from runners.predictor import Predictor
from utils.stitcher import Stitcher2d

from utils.arg_parser import Argments


FL_TYPES = ["mem", "act", "mito", "lipid", "nuc", "oli"]


class Ri2Fl(object):
    def __init__(self, arg_path, cmd_args):
        self.arg = Argments(f"scripts/{arg_path}", cmd_args)
        self.dataset_path = self.arg["path/dataset"]
        self.save_path = self.arg["path/save_path"]
        p = re.compile("\w+")
        self.fl_list = p.findall(self.arg["setup/fl_list"])
        for fl in self.fl_list:
            assert (
                fl in FL_TYPES
            ), f"'{fl}' is not valid. All elements should be one of {FL_TYPES}."

    def predict(self, fl_type):
        assert (
            fl_type in self.fl_list
        ), f"'{fl_type}' is not valid. All elements should be one of {FL_TYPES}."
        self.arg["setup/fl_type"] = fl_type
        self.arg.reset()
        setup = self.arg["setup"]
        loader = SlideInferLoader(
            self.dataset_path,
            setup["batch_size"],
            setup["cpus"],
        )
        stitcher = Stitcher2d(
            setup["cropped_depth"],
            setup["patch_size"],
            setup["zoomed_size"][0],
            lambda x: x,
        )
        runner = Predictor(
            self.arg["model"],
            loader,
            stitcher,
            self.save_path,
            fl_type,
            setup["num_drop"],
            setup["num_tta"],
        )

        runner.infer_patch()
        dist.barrier()
        runner.stitch()
        dist.barrier()

    def predict_all(self):
        for fl_type in self.fl_list:
            self.predict(fl_type)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("yaml")
    argparser.add_argument("--local_rank", default=0, type=int)
    cmd_args = argparser.parse_args()
    ri2fl = Ri2Fl(f"{cmd_args.yaml}.yaml", cmd_args)
    ri2fl.predict_all()
    dist.destroy_process_group()
