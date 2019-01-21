import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.measure import compare_psnr
from multiprocessing import Pool
import os
from timeit import default_timer as timer

def clip(img):
    img = np.minimum(np.ones(img.shape), img)
    img = np.maximum(np.zeros(img.shape), img)
    return img

def readImg(path):
    return io.imread(path, as_gray = True).astype('float64')

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

def denoiseRow(img, x, left_y, right_y, K, L, sig):
    # print(x)
    return (x, left_y, right_y, 
            [denoise(img, x, y, K, L, sig) for y in range(left_y, right_y)])

def denoiseImage(img, K, L, sig):
    global outImg

    outImg = np.copy(img)
    width, height = img.shape
    halfL = L // 2

    def denoiseRowCallback(result):
        global outImg

        x, y_left, y_right, data = result
        outImg[x, y_left:y_right] = data

    # parallel
    progress = [pool.apply_async(denoiseRow, (img, x, halfL, height - halfL, K, L, sig,), callback=denoiseRowCallback) for x in range(halfL, width - halfL)]
    for each in progress:
        each.wait()

    # non-parallel:
    # for x in range(halfL, width - halfL):
    #     print(x)
    #     for y in range(halfL, height - halfL):
    #         outImg[x, y] = denoise(img, x, y, K, L, sig)

    return outImg

############ main #################
if __name__ == '__main__':

    pool = Pool(os.cpu_count() - 3)

    originalImage = readImg('campus/Lena512.png')

    v = 20/255.0
    noisedImage = readImg('campus/Lena512_noi_s25.png')

    K = 5
    L = 21
    sig1 = v #0.015

    print(str("sig1").ljust(23), str("coef").ljust(20), str("psnr2").ljust(20), str("seconds").ljust(20))

    for sig1 in np.logspace(-3, -1, 10):
    # for sig1 in [v]:
        for coef in np.logspace(-2, 1/3, 10):
        # for coef in [0.35]:

            start = timer()

            stage1 = denoiseImage(noisedImage, K, L, sig1)
            io.imsave('campus/Lena512_denoised_1_step.png', stage1)

            sig2 = coef * np.sqrt(sig1 - np.mean((stage1 - noisedImage)**2))
            # print('sig2 = ', sig2)

            stage2 = denoiseImage(stage1, K, L, sig2)
            io.imsave('campus/Lena512_denoised_2_step.png', stage2)

            end = timer()

            psnr1 = compare_psnr(stage1, originalImage)
            psnr2 = compare_psnr(stage2, originalImage)
            print(str(sig1).ljust(23), str(coef).ljust(20), str(psnr2).ljust(20), str(end - start).ljust(20))