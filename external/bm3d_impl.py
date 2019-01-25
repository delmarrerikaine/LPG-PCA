# https://github.com/ericmjonas/pybm3d
# https://github.com/ericmjonas/pybm3d/issues/10

# runs only on Mac and Linus with Python 3.6!

import pybm3d


def denoise(image, noise_std_dev):
    return pybm3d.bm3d.bm3d(255 * image, 255 * noise_std_dev) / 255
