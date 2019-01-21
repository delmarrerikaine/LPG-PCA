import numpy as np
import math
from skimage import io
from skimage.measure import compare_psnr
from skimage.util import random_noise
import os

import mf_impl
import bm3d_impl
import nlm_impl

noise_variance = 40.0 / 255.0

algorithms = [mf_impl, bm3d_impl, nlm_impl]
psnrs = np.zeros(len(algorithms))

def getNoisedImage(oI, v):
    np.random.seed(42)
    noise = np.random.normal(size = oI.shape)
    noise = noise/np.sqrt(np.power(noise, 2).mean())
    noisedImage = oI + v*noise
    return noisedImage
def clip(img):
    img = np.minimum(np.ones(img.shape), img)
    img = np.maximum(np.zeros(img.shape), img)
    return img
def readImg(path):
    return io.imread(path, as_gray = True).astype('float64')/255.0
def showImg(img, name):
    print(name)
    img = clip(img)
    io.imshow((img*255.0).astype('uint8'))

img_names = os.listdir(os.path.join(os.path.dirname(__file__), 'images/'))
print()
print("                     [mf          bm3d        nlm        ]")
for img_name in img_names:
    pure_image = readImg('images/' + img_name)

    # noised_image = random_noise(pure_image, mode='gaussian', var=noise_variance)
    noised_image = getNoisedImage(pure_image, noise_variance)

    for i, algorithm in enumerate(algorithms):
        denoised_image = algorithm.denoise(noised_image, noise_variance)
        psnr = compare_psnr(pure_image, denoised_image)
        psnrs[i] = psnr

    print(img_name.ljust(20), psnrs)

print()