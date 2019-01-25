import numpy as np
import pandas as pd
from skimage import io
import skimage.measure as measure
import os
from lpg_pca_impl import denoise

def getNoisedImage(originalImage, variance):
    # return random_noise(originalImage, mode='gaussian', var=variance)
    np.random.seed(42)
    noise = np.random.normal(size = originalImage.shape)
    noise = noise/np.sqrt(np.power(noise, 2).mean())
    noisedImage = originalImage + variance*noise
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

def saveImg(img, path):
    img = clip(img)
    io.imsave(path, (img*255.0).astype('uint8'))

def compare_psnr(img1, img2):
    return measure.compare_psnr(img1, img2)

def compare_ssim(img1, img2):
    return measure.compare_ssim(img1, img2)

def generate_images(img_name='mri'):
    experiments_folder = 'experiments'
    noise_variances = [10, 20, 30, 40]

    for noise_variance in noise_variances:
        corrected_noise_variance = noise_variance / 255.0

        original_img = readImg(os.path.join('images', img_name + '.png'))

        noised_img = getNoisedImage(original_img, corrected_noise_variance)

        noised_file_name = img_name + '_noised_' + str(noise_variance) + '.png'
        saveImg(noised_img, os.path.join(experiments_folder, noised_file_name))
        print(noised_file_name + ' started.')

        denoised_img = denoise(noised_img, noise_variance)

        denoised_file_name = img_name + '_denoised_' + str(noise_variance) + '.png'
        saveImg(denoised_img, os.path.join(experiments_folder, denoised_file_name))
        print(denoised_file_name + ' finished.')

        print("noised PSNR: " + str(compare_psnr(original_img, noised_img)) + ", SSIM: " + str(compare_ssim(original_img, noised_img)))
        print("denoised PSNR: " + str(compare_psnr(original_img, denoised_img)) + ", SSIM: " + str(compare_ssim(original_img, denoised_img)))

def generate_latex_tables():
    df = pd.read_csv('data.csv')
    df = df.round(2)

    image_texts = np.array([])

    temp_directory = os.path.join(os.path.dirname(__file__), 'temp')
    if not os.path.exists(temp_directory):
        os.makedirs(temp_directory)

    for image_name in list(set(df['image_name'])):
        image_df = df[df['image_name'] == image_name]
        image_df['denoise_lpg_pca'] = image_df['denoise_psnr_lpg_pca'].map(str) + '(' +  image_df['denoise_ssim_lpg_pca'].map(str) + ')'
        image_df['denoise_mf'] = image_df['denoise_psnr_mf'].map(str) + '(' +  image_df['denoise_ssim_mf'].map(str) + ')'
        image_df['denoise_nlm'] = image_df['denoise_psnr_nlm'].map(str) + '(' +  image_df['denoise_ssim_nlm'].map(str) + ')'
        image_df['denoise_bm3d'] = image_df['denoise_psnr_bm3d'].map(str) + '(' +  image_df['denoise_ssim_bm3d'].map(str) + ')'
        image_df = image_df[['sigma', 'denoise_lpg_pca', 'denoise_mf', 'denoise_nlm', 'denoise_bm3d']]
        image_df['sigma'] = image_df['sigma'].map(int)
        image_df.columns = ['sigma', 'LPG-PCA', 'MF', "NLM", 'BM3D']

        path = os.path.join(temp_directory, image_name + '.tex')
        image_df.to_latex(path, index=False, column_format='lrrrr')

        with open(path, 'r') as file:
            image_text = file.read()
            image_text = image_text.replace(' ', '').replace(r'\toprule',r'\toprule &&' + image_name + r'\\ \midrule')
            image_text = r'\noindent\begin{minipage}{.5\linewidth}' + '\n' + image_text + '\n' + r'\end{minipage}'
            image_text = image_text.replace('\n\n', '\n').replace('sigma&', '$\\sigma$&')
            image_texts = np.append(image_texts, image_text)
        
        os.remove(path)

    result = '\n'.join(image_texts)

    filename = 'tables.tex'
    with open(filename, "w+") as file:
        file.write(result)

    if(len(os.listdir(temp_directory))) == 0:
        os.rmdir(temp_directory)