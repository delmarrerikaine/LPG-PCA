import lpg_pca_impl as lpg_pca
from skimage.measure import compare_psnr
from util import readImg
import os

noised_img = readImg(os.path.join('temp', 'Lena512.png'))
original_img = readImg(os.path.join('temp', 'Lena512_noi_s25.png'))

denoised_img = lpg_pca.denoise(noised_img, 20 / 255, log=True)

print("PSNR: ", compare_psnr(denoised_img, original_img))