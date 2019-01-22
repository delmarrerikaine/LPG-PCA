import lpg_pca_impl as lpg_pca
from skimage.measure import compare_psnr
from util import readImg, getNoisedImage, saveImg
import os


if __name__ == '__main__':
#    noised_img = readImg(os.path.join('temp', 'Lena512_noi_s25.png'))

    noise_variances = [10.0 / 255.0, 20.0 / 255.0, 30.0 / 255.0, 40.0 / 255.0]
    Ks = [3, 5, 7]
    results = []
    for noise_variance in noise_variances:
        for K in Ks:
            name = 'lenaAfter_'+ str(noise_variance) + '_' + str(K) + '.png'
            print("Start: ", name)
            original_img = readImg(os.path.join('images', 'Lena512.png'))

            noised_img = getNoisedImage(original_img, noise_variance)
            saveImg(noised_img, os.path.join('experiments', 'noised_' + name))
            noise_psnr = compare_psnr(original_img, noised_img)
            print("Noise PSNR: ", noise_psnr)

            denoised_img = lpg_pca.denoise(noised_img, noise_variance, K = K, log=False)
            saveImg(denoised_img, os.path.join('experiments', 'denoise_' + name))
            denoise_psnr = compare_psnr(original_img, denoised_img)
            print("Denoise PSNR: ", denoise_psnr)
            results.append([name, noise_psnr, denoise_psnr])
    print(results)