import random
import numpy as np
from scipy.ndimage import gaussian_filter, zoom, rotate, map_coordinates


def identity(x, mag):
    return x


def gamma_correction(x, mag):
    min_, max_ = x.min(), x.max()
    x_ = (x - min_) / (max_ - min_)
    x_ = x_ ** mag
    x_ = x_ * (max_ - min_) + min_
    return x_


def posterize(x, mag):
    mag = 10 ** (mag // 5 + 1)
    x = (x * mag).astype(np.int).astype(np.float) / mag
    return x


def additive_gaussian(x, mag):
    noise = np.random.random(x.shape) * mag
    return (x + noise).clip(x.min(), x.max())


def blur(x, mag):
    return gaussian_filter(x, mag).clip(x.min(), x.max())


def sharpness(x, mag):
    blur = gaussian_filter(x, 3)
    blur_2 = gaussian_filter(blur, 1)
    sharpened = blur + mag * (blur - blur_2)
    return sharpened.clip(x.min(), x.max())


def brightness(x, mag):
    return (x * mag).clip(x.min(), x.max())


def elastic_deform(x, y, mag):
    alpha, sigma = mag, 3 - mag

    shape = x.shape
    random_state = np.random.RandomState(None)

    dx = (
        gaussian_filter(
            (random_state.rand(shape[1], shape[2]) * 2 - 1),
            sigma,
            mode="constant",
            cval=0,
        )
        * alpha
    )
    dy = (
        gaussian_filter(
            (random_state.rand(shape[1], shape[2]) * 2 - 1),
            sigma,
            mode="constant",
            cval=0,
        )
        * alpha
    )

    _x, _y = np.meshgrid(np.arange(shape[1]), np.arange(shape[2]), indexing="ij")
    indices = np.reshape(_x + dx, (-1, 1)), np.reshape(_y + dy, (-1, 1))

    ret_x = np.zeros(shape)
    ret_y = np.zeros(shape)
    for i in range(shape[0]):
        ret_x[i] = map_coordinates(x[i], indices, order=1, mode="reflect").reshape(
            shape[1:]
        )
        ret_y[i] = map_coordinates(y[i], indices, order=0, mode="reflect").reshape(
            shape[1:]
        )
    return ret_x, ret_y


def resizing_crop(x, y, mag):
    resized_x = zoom(x, (1, mag, mag), order=1)
    resized_y = zoom(y, (1, mag, mag), order=1)
    offset = resized_x.shape[-1] - x.shape[-1]
    beg = offset // 2
    end = beg + x.shape[-1]
    return (
        resized_x[:, beg:end, beg:end],
        resized_y[:, beg:end, beg:end],
    )


def resizing_crop_x(x, y, mag):
    resized_x = zoom(x, (1, 1, mag), order=1)
    resized_y = zoom(y, (1, 1, mag), order=1)
    offset = resized_x.shape[-1] - x.shape[-1]
    beg = offset // 2
    end = beg + x.shape[-1]
    return resized_x[:, :, beg:end], resized_y[:, :, beg:end]


def resizing_crop_y(x, y, mag):
    resized_x = zoom(x, (1, mag, 1), order=1)
    resized_y = zoom(y, (1, mag, 1), order=1)
    offset = resized_x.shape[-2] - x.shape[-2]
    beg = offset // 2
    end = beg + x.shape[-2]
    return resized_x[:, beg:end, :], resized_y[:, beg:end, :]


def flip(x, y, mag):
    if mag % 3 == 0:
        ret = np.array(x)[..., ::-1], np.array(y)[..., ::-1]
    elif mag % 3 == 1:
        ret = np.array(x)[:, ::-1], np.array(y)[:, ::-1]
    elif mag % 3 == 2:
        ret = (
            np.array(x)[:, ::-1, ::-1],
            np.array(y)[:, ::-1, ::-1],
        )
    return ret


def _rotate(x, y, mag):
    _, h, w = x.shape
    rot_x = rotate(x, mag, axes=(1, 2), order=1, reshape=False)
    rot_y = rotate(y, mag, axes=(1, 2), order=0, reshape=False)
    __, _h, _w = rot_x.shape
    c_h = _h // 2 - h // 2
    c_w = _w // 2 - w // 2
    new_x = rot_x[:, c_h : c_h + h, c_w : c_w + w]
    new_y = rot_y[:, c_h : c_h + h, c_w : c_w + w]
    return new_x, new_y


def augment_list():
    aug_list = [
        (identity, 0.0, 1.0),
        (gamma_correction, 0.5, 1.5),
        (posterize, 0, 30),
        (additive_gaussian, 0.0, 0.3),
        (blur, 0.0, 0.3),
        (sharpness, 85.0, 100.0),
        (brightness, 0.7, 1.3),
    ]

    return aug_list


def default_list():
    augment_list = [
        (resizing_crop, 1.0, 1.3),
        (resizing_crop_x, 1.0, 1.3),
        (resizing_crop_y, 1.0, 1.3),
        (flip, 0.0, 30.0),
        (elastic_deform, 0.0, 3.0),
    ]

    return augment_list


class RandAugment(object):
    def __init__(self, n, m=None):
        self.n = n
        self.m = m
        self.aug_list = augment_list()
        self.default_list = default_list()

    def __call__(self, x, y):
        ops = random.choices(self.aug_list, k=self.n)
        for op, minval, maxval in ops:
            if self.m:
                m = self.m
            else:
                m = np.random.randint(0, 30)
            val = (float(m) / 30) * float(maxval - minval) + minval
            x = op(x, mag=val)
        for op, minval, maxval in self.default_list:
            if self.m:
                m = self.m
            else:
                m = np.random.randint(0, 30)
            val = (float(m) / 30) * float(maxval - minval) + minval
            x, y = op(x, y, mag=val)
        return x, y


if __name__ == "__main__":
    a = np.random.rand(80, 160, 160)
    b = a * 100

    b_pos, _ = elastic_deform(b, b, 3)
    print(b_pos.min(), b_pos.max())
