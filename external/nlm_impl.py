# http://scikit-image.org/docs/dev/auto_examples/filters/plot_nonlocal_means.html

import numpy as np
from skimage.restoration import denoise_nl_means

def denoise(image, noise_std_dev):
    return denoise_nl_means(image, sigma=noise_std_dev, patch_size=5, patch_distance=6, multichannel=False)