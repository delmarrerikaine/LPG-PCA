import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.measure import compare_psnr
from multiprocessing import Pool

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

def getNoisedImage(oI, v):
    np.random.seed(42)
    noise = np.random.normal(size = originalImage.shape)
    noise = noise/np.sqrt(np.power(noise, 2).mean())
    noisedImage = oI + v*noise
    return noisedImage

def denoise(img, x, y, K, L, sig):
    def getBlock(x, y):#img, halfK, x, y
        return img[x - halfK: x + halfK + 1, y - halfK: y + halfK + 1]
    def mse(block):
        return np.mean((block - target)**2)
    halfK = K//2
    halfL = L//2
    # Dimension of each block vector (= number of rows in the training matrix)
    m = K**2
    
    # Number of columns in the training matrix
    n = m * 8 + 1
    
    # Block centered around x,y
    target = getBlock(x, y)
    
    # Assemble a pool of blocks.
    blocks = []
    rng = halfL - halfK
    for ty in range(y-rng, y+rng+1):
        for tx in range(x-rng, x+rng+1):
            # Exclude target
            if tx == x and ty == y:
                continue
            block = getBlock(tx, ty)
            blocks.append(block)
    
    blocks.sort(key = mse)

    # Construct the training matrix with the target and the best blocks reshaped into columns.
    trainingMatrix = np.hstack((target.reshape(m, 1, order='F'), 
                                np.transpose([np.array(block).reshape(m, order='F') for block in blocks[:n]])))

    mean = trainingMatrix.mean(axis=1)
    trainingMatrix = trainingMatrix - mean.reshape(m,1)
    noiseCov = sig**2 * np.eye(m,m)
    inputCov = (trainingMatrix @ trainingMatrix.T)/n
    eigvectors = np.linalg.eig(inputCov)[1]
    PX = eigvectors.T
    
    transInput = PX @ trainingMatrix
    
    transNoiseCov = PX @ noiseCov @ PX.T
    transInputCov = (transInput @ transInput.T)/n
    transDenoisedOutCov = np.maximum(np.zeros(transInputCov.shape), transInputCov - transNoiseCov)
    
    shrinkCoef = np.diag(transDenoisedOutCov)/(np.diag(transDenoisedOutCov) + np.diag(transInputCov))
    Y1 = transInput[:,0] * shrinkCoef
    X1 = PX.T @ Y1 + mean
    return X1[m//2]

# def denoiseRow(img, x, left_y, right_y, K, L, sig, outImg):
#     print(x)
#     for y in range(left_y, right_y):
#         outImg[x, y] = denoise(img, x, y, K, L, sig)

def denoiseImage(img, K, L, sig):
    outImg = np.copy(img)
    width, height = img.shape
    halfL = L // 2

    # parallel
    # done = [pool.apply_async(denoiseRow, (img, x, halfL, height - halfL, K, L, sig, outImg,)) for x in range(halfL, width - halfL)]
    # for each in done:
    #     each.wait()

    # non-parallel:
    for x in range(halfL, width - halfL):
        print(x)
        for y in range(halfL, height - halfL):
            outImg[x, y] = denoise(img, x, y, K, L, sig)

    return outImg

############ main #################
# if __name__ == '__main__':
    # pool = Pool(4)

originalImage = readImg('campus/campus_greyscale.png')

v = 20/255.0
noisedImage = readImg('campus/campus_greyscale_noise.jpg')

K = 5
L = 21
sig1 = v #0.015

stage1 = denoiseImage(noisedImage, K, L, sig1)
print("PSNR1:", compare_psnr(stage1, originalImage))
io.imsave('campus/campus_greyscale_denoised_1_step.jpg', stage1)

sig2 = 0.35 * np.sqrt(sig1 - np.mean((stage1 - noisedImage)**2))
print("Sig2: ", sig2)

stage2 = denoiseImage(stage1, K, L, sig2)

print("PSNR2:", compare_psnr(stage2, originalImage))
io.imsave('campus/campus_greyscale_denoised_2_step.jpg', stage2)