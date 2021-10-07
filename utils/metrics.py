import torch


class GDLoss3d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, pred, target):
        dx_pred = pred[..., 1:] - pred[..., :-1]
        dy_pred = pred[..., 1:, :] - pred[..., :-1, :]
        dz_pred = pred[..., 1:, :, :] - pred[..., :-1, :, :]

        dx_target = target[..., 1:] - target[..., :-1]
        dy_target = target[..., 1:, :] - target[..., :-1, :]
        dz_target = target[..., 1:, :, :] - target[..., :-1, :, :]

        mse_dx = self.mse(dx_pred.abs(), dx_target.abs())
        mse_dy = self.mse(dy_pred.abs(), dy_target.abs())
        mse_dz = self.mse(dz_pred.abs(), dz_target.abs())

        return (mse_dx + mse_dy + mse_dz) / 3.0


class PCC(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target):
        x = input
        y = target

        non_batch_axis = list(range(x.dim()))[1:]
        std_x = x.std(non_batch_axis, keepdim=True)
        std_y = y.std(non_batch_axis, keepdim=True)

        mean_x = x.mean(non_batch_axis, keepdim=True)
        mean_y = y.mean(non_batch_axis, keepdim=True)

        vx = (x - mean_x) / (std_x + 0.0001)
        vy = (y - mean_y) / (std_y + 0.0001)
        _pcc = (vx * vy).mean(non_batch_axis)

        if self.reduction == "mean":
            _pcc = _pcc.mean()
        elif self.reduction == "sum":
            _pcc = _pcc.sum()
        return _pcc
