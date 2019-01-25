import lpg_pca_impl as lpg_pca
import external.mf_impl as mf
import external.nlm_impl as nlm
from util import readImg, getNoisedImage, saveImg, compare_psnr, compare_ssim
import os
import pandas as pd

data_file_name = r'data.csv'
images_folder = r'images'

if __name__ == '__main__':

    noise_variances = [10.0, 20.0, 30.0, 40.0]
    results = []

    img_names = os.listdir(os.path.join(os.path.dirname(__file__), images_folder))

    for img_name in img_names:

        name = os.path.splitext(img_name)[0]

        # to skip some images
        # if name in ['barbara', 'boat', 'cameraman', 'couple', 'fingerprint', 'hill', 'house', 'lena', 'man', 'montage', 'peppers']:
        #     continue

        original_img = readImg(os.path.join(images_folder, img_name))

        for noise_variance in noise_variances:
            corrected_noise_variance = noise_variance / 255

            noised_img = getNoisedImage(original_img, corrected_noise_variance)

            noise_psnr = compare_psnr(original_img, noised_img)
            noise_ssim = compare_ssim(original_img, noised_img)

            denoised_img_lpg_pca = lpg_pca.denoise(noised_img, corrected_noise_variance)
            denoised_img_mf = mf.denoise(noised_img, corrected_noise_variance)
            denoised_img_nlm = nlm.denoise(noised_img, corrected_noise_variance)

            denoise_psnr_lpg_pca = compare_psnr(original_img, denoised_img_lpg_pca)
            denoise_psnr_mf = compare_psnr(original_img, denoised_img_mf)
            denoise_psnr_nlm = compare_psnr(original_img, denoised_img_nlm)

            denoise_ssim_lpg_pca = compare_ssim(original_img, denoised_img_lpg_pca)
            denoise_ssim_mf = compare_ssim(original_img, denoised_img_mf)
            denoise_ssim_nlm = compare_ssim(original_img, denoised_img_nlm)

            row_results = [name, noise_variance, noise_psnr, noise_ssim, denoise_psnr_lpg_pca, denoise_ssim_lpg_pca, denoise_psnr_mf, denoise_ssim_mf, denoise_psnr_nlm, denoise_ssim_nlm]
            print(row_results)
            results.append(row_results)

            cols = ['image_name', 'sigma', 'noise_psnr', 'noise_ssim', 'denoise_psnr_lpg_pca', 'denoise_ssim_lpg_pca', 'denoise_psnr_mf', 'denoise_ssim_mf', 'denoise_psnr_nlm', 'denoise_ssim_nlm']
            df_prev = pd.DataFrame([], columns=cols)

            if(os.path.isfile(data_file_name)):
                df_prev = pd.read_csv(data_file_name)

            df_add = pd.DataFrame([row_results], columns=cols)
            df_full = df_prev.append(df_add)

            df_full.to_csv(data_file_name, index=False)
