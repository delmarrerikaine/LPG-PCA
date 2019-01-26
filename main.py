from util import readImg, getNoisedImage, compare_psnr
from lpg_pca_impl import denoise
import os
import cProfile
import timeit

def denoise_external():
    img = readImg(os.path.join('images', 'lena.png'))

    noised_img = getNoisedImage(img, 20/255)

    denoised_img = denoise(noised_img, 20/255)

    print("PSNR:", compare_psnr(img, denoised_img))

if __name__ == '__main__':
    
    # cProfile.run('denoise_external()')
    denoise_external()
    print(timeit.timeit(stmt=denoise_external, number=5)/5)
    