from lpg_pca_impl import denoise
from util import readImg, saveImg, getNoisedImage
import os

img_name = 'fingerprint'
noise_variances = [10, 20, 30, 40]

for noise_variance in noise_variances:
    corrected_noise_variance = noise_variance / 255.0

    original_img = readImg(os.path.join('images', img_name + '.png'))

    noised_img = getNoisedImage(original_img, corrected_noise_variance)

    noised_file_name = img_name + '_noised_' + str(noise_variance) + '.png'
    saveImg(noised_img, os.path.join('generated', noised_file_name))
    print(noised_file_name + ' started.')

    denoised_img = denoise(noised_img, noise_variance)

    denoised_file_name = img_name + '_denoised_' + str(noise_variance) + '.png'
    saveImg(denoised_img, os.path.join('generated', denoised_file_name))
    print(denoised_file_name + ' finished.')