import numpy as np


def _spline_window(window_size, power=2):
    intersection = window_size // 4
    tri = 1. - np.abs((window_size - 1) / 2. - np.arange(0, window_size)) / ((window_size - 1) / 2.)
    wind_outer = np.power(tri * 2, power) / 2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - np.abs(2 * (tri - 1)) ** power / 2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    wind = wind_inner + wind_outer
    return wind


class Stitcher2d(object):
    """
    Stitches, blends and unpads output patches.


    Args :
        img_size (tuple) : a shape of an original image (D, H, W)
        patch_size (int) : a size of a patch
        zoomed_size (int) : a size of zoomed image
        post_op (function) : a function for post processing

    Example::
        >>> stitcher = Stitcher2d(cropped_depth, self.patch_size, self.zoomed_size, self.post_op)
    """
    def __init__(self, depth, patch_size, zoomed_size, post_op):
        self.patch_size = patch_size
        self.pad_size = patch_size // 2
        self.post_op = post_op
        self.full_index = ((zoomed_size - self.patch_size) / (self.patch_size // 2) + 3) ** 2
        self.canvas_size = (depth, zoomed_size + self.pad_size * 2, zoomed_size + self.pad_size * 2)
        self.window_2d = self._window_2d(patch_size, 1)

    def stitch(self, patches):
        stitched_img = {i: np.zeros(self.canvas_size) for i in list(patches.values())[0]}

        for coord, imgs in patches.items():
            y, x = coord
            for img_type, img in imgs.items():
                stitched_img[img_type][:, y: y + self.pad_size * 2, x: x + self.pad_size * 2] += img * self.window_2d

        for k, v in stitched_img.items():
            unpad = self.pad_size
            stitched_img[k] = self.post_op(v[:, unpad:-unpad, unpad:-unpad])

        del patches

        return stitched_img

    def _window_2d(self, window_size=128, power=2):
        """
            Make a 1D window function, then infer and return a 2D window function.
            Done with an augmentation, and self multiplication with its transpose.
            Could be generalized to more dimensions.
            """
        wind = _spline_window(window_size, power)
        wind = np.expand_dims(np.expand_dims(wind, 1), 2)
        wind = wind * wind.transpose(1, 0, 2)
        return wind.transpose(2, 0, 1)
