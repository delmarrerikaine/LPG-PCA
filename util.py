import numpy as np
from skimage import io
import skimage.measure as measure

def getNoisedImage(oI, v):
    # return random_noise(oI, mode='gaussian', var=v)
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

def compare_psnr(img1, img2):
    return measure.compare_psnr(img1, img2)