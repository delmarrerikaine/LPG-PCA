import numpy as np
import math
from skimage import io
from skimage.util import random_noise
import os

from util import readImg, compare_psnr, getNoisedImage

import mf_impl
import bm3d_impl
import nlm_impl
import lpg_pca_impl

if __name__ == '__main__':

    noise_variance = 40.0 / 255.0

    algorithms = [mf_impl, bm3d_impl, nlm_impl, lpg_pca_impl]
    psnrs = np.zeros(len(algorithms))

    img_names = os.listdir(os.path.join(os.path.dirname(__file__), 'images'))
    print()
    print("                     [mf          bm3d        nlm        lpg-pca      ]")
    for img_name in img_names:
        pure_image = readImg(os.path.join('images', img_name))

        noised_image = getNoisedImage(pure_image, noise_variance)

        for i, algorithm in enumerate(algorithms):
            denoised_image = algorithm.denoise(noised_image, noise_variance)
            psnr = compare_psnr(pure_image, denoised_image)
            psnrs[i] = psnr

        print(img_name.ljust(20), psnrs)

    print()