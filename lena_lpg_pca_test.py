import lpg_pca_impl as lpg_pca
from util import readImg, compare_psnr, getNoisedImage, saveImg

if __name__ == '__main__':

    img = readImg('images/lena.png')
    noised_img = getNoisedImage(img, 20/255)
    saveImg(noised_img, 'noised.png')

    denoised_img = lpg_pca.denoise(noised_img, 20/255, log=True)

    saveImg(denoised_img, 'denoised.png')

    print(noised_img)
    print(denoised_img)

    print(compare_psnr(img, noised_img))
    print(compare_psnr(img, denoised_img))