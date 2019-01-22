import numpy as np
from multiprocessing import Pool
import os

def _denoise_pixel(img, x, y, K, L, sig):
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
    dim1, dim2 = img.shape
    blocks = []
    rng = halfL - halfK
    for ty in range(max(K, y-rng), min(y+rng+1, dim1-K)):
        for tx in range(max(K, x-rng), min(x+rng+1, dim2-K)):
            # Exclude target
            if tx == x and ty == y:
                continue
            block = getBlock(tx, ty)
            blocks.append(block)
    
    blocks.sort(key = mse)

    # Construct the training matrix with the target and the best blocks reshaped into columns.
    if len(blocks)<n:
        n = len(blocks)
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

def _denoise_row(img, x, left_y, right_y, K, L, sig, log):
    if log:
        print(x)
    return (x, left_y, right_y, 
            [_denoise_pixel(img, x, y, K, L, sig) for y in range(left_y, right_y)])

def _denoise_image(img, K, L, sig, log):
    global outImg

    outImg = np.copy(img)
    width, height = img.shape
    halfL = L // 2
    halfK = K // 2

    def denoiseRowCallback(result):
        global outImg

        x, y_left, y_right, data = result
        outImg[x, y_left:y_right] = data

    global pool

    # parallel
    progress = [pool.apply_async(_denoise_row, (img, x, halfK, height - halfK, K, L, sig, log,), callback=denoiseRowCallback) for x in range(halfK, width - halfK)]
    for each in progress:
        each.wait()

    # non-parallel:
    # for x in range(0, width):
    #     if log:
    #         print(x)
    #     for y in range(0, height):
    #         outImg[x, y] = _denoise_pixel(img, x, y, K, L, sig)

    return outImg

def denoise(noised_img, sig1, K=7, L=21, log=False):
    global pool

    pool = Pool(os.cpu_count() - 2)

    stage1 = _denoise_image(noised_img, K, L, sig1, log)

    sig2 = 0.35 * np.sqrt(sig1 - np.mean((stage1 - noised_img)**2))
    if log:
        print('sig2 = ', sig2)

    stage2 = _denoise_image(stage1, K, L, sig2, log)

    return stage2