import os
import argparse
import torch
from pathlib import Path
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from loaders.slideloader import PatchLoader
from utils.metrics import GDLoss3d, PCC
import numpy as np


@torch.no_grad()
def reduce_tensor(tensor, average=False):
    world_size = dist.get_world_size()
    if world_size < 2:
        return tensor
    temp = tensor.clone()
    dist.all_reduce(temp)
    if dist.get_rank() == 0 and average:
        temp /= world_size
    return temp


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_root", type=str)
    argparser.add_argument("--fl_type", default="nuc", type=str)
    argparser.add_argument("--phase", default="train", type=str)
    argparser.add_argument("--gpus", default="0", type=str)
    argparser.add_argument("--local_rank", default=0, type=int)
    argparser.add_argument("--resume", default="none", type=str)
    argparser.add_argument("--save_path", default="outs/nuc", type=str)
    cmd_args = argparser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = cmd_args.gpus

    save_path = cmd_args.save_path
    Path(save_path).mkdir(parents=True, exist_ok=True)
    world_size = len(cmd_args.gpus.replace(",", "").replace("'", ""))
    torch.cuda.set_device(cmd_args.local_rank)
    sharedfile_path = "env://"
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=sharedfile_path,
        world_size=world_size,
        rank=cmd_args.local_rank,
    )

    data_loader = PatchLoader(cmd_args.data_root, cmd_args.fl_type, 1, 4)

    model = torch.jit.load("models/act.pth").cuda()
    optimizer = torch.optim.Adam(model.parameters(), 0.001)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", 0.2, 30)

    mse_fn = torch.nn.MSELoss()
    gdl_fn = GDLoss3d()
    model = DistributedDataParallel(model, [cmd_args.local_rank])

    if Path(cmd_args.resume).exists():
        ckp = torch.load(cmd_args.resume, map_location="cpu")
        model.load_state_dict(ckp["param"])
        lr_scheduler.load_state_dict(ckp["lr_scheduler"])
        start_epoch = ckp["epoch"] + 1
    else:
        for name, m in model.named_parameters():
            if "bn.weight" in name:
                torch.nn.init.constant_(m, 1)
            elif "bias" in name:
                torch.nn.init.constant_(m, 0)
            else:
                torch.nn.init.kaiming_normal_(m)
        start_epoch = 0

    if cmd_args.local_rank == 0:
        print("\t======================")
        print("\tEpoch\tLoss\tPCC")
        print("\t======================")
    for epoch in range(start_epoch, 200):
        losses = []
        model.train()
        for ri, fl in data_loader.load("train"):
            ri, fl = ri.cuda(), fl.cuda()
            output = model(ri)

            mse_loss = mse_fn(output, fl)
            gdl_loss = gdl_fn(output, fl)
            loss = mse_loss + gdl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses += [reduce_tensor(loss, True).item()]

        model.eval()
        pcc_max = 0
        with torch.no_grad():
            pcc_fn = PCC()
            pcces = []
            for ri, fl in data_loader.load("val"):
                ri, fl = ri.cuda(), fl.cuda()
                output = model(ri)

                pcc = pcc_fn(output, fl)
                pcces += [reduce_tensor(pcc, True).item()]
            if cmd_args.local_rank == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "param": model.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    f"{save_path}/model.pth",
                )

                if pcc_max < np.mean(pcces):
                    torch.save(
                        {
                            "epoch": epoch,
                            "param": model.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                        },
                        f"{save_path}/best.pth",
                    )
                print(f"\t{epoch}\t{np.mean(losses):.4f}\t{np.mean(pcces)*100:.2f}")
                pcc_max = np.mean(pcces)


if __name__ == "__main__":
    main()
