import numpy as np
from multiprocessing import Pool
import os
import multiprocessing
import logging

def getBlock(leftX, rightX, leftY, rightY):
    global img
        
    return img[leftX: rightX, leftY: rightY]

def _denoise_pixel(x, y, K, L, sig):
    global img

    def mse(block):
        return np.mean((block - target)**2)
    halfK = K//2
    halfL = L//2
    # Dimension of each block vector (= number of rows in the training matrix)
    m = K**2
    
    # Number of columns in the training matrix
    n = m * 8 + 1

    # print(getBlock(100, 100))
    
    # Block centered around x,y
    target = getBlock(x - halfK, x + halfK + 1, y - halfK, y + halfK + 1)
    
    # Assemble a pool of blocks.
    dim1, dim2 = img.shape
    blocks = []
    rng = halfL - halfK
    for ty in range(max(K, y-rng), min(y+rng+1, dim1-K)):
        for tx in range(max(K, x-rng), min(x+rng+1, dim2-K)):
            # Exclude target
            if tx == x and ty == y:
                continue
            block = getBlock(tx - halfK, tx + halfK + 1, ty - halfK, ty + halfK + 1)
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

    del blocks
    del trainingMatrix
    return X1[m//2]

def _denoise_row(x, left_y, right_y, K, L, sig, log):
    if log:
        print(x)
    return (x, left_y, right_y, 
            [_denoise_pixel(x, y, K, L, sig) for y in range(left_y, right_y)])

def _denoise_image(K, L, sig, log):
    global outImg
    global img

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
    progress = [pool.apply_async(_denoise_row, (x, halfK, height - halfK, K, L, sig, log,), callback=denoiseRowCallback) for x in range(halfK, width - halfK)]
    for each in progress:
        each.wait()

    # non-parallel:
    # for x in range(halfK, width - halfK):
    #     if log:
    #         print(x)
    #     for y in range(halfK, height - halfK):
    #         outImg[x, y] = _denoise_pixel(x, y, K, L, sig)

    return outImg

def denoise(noised_img, sig1, K=3, L=21, log=False):
    global pool
    global img

    multiprocessing.log_to_stderr()
    logger = multiprocessing.get_logger()
    logger.setLevel(logging.INFO)

    img = np.copy(noised_img)

    pool = Pool(os.cpu_count() - 1)

    stage1 = _denoise_image(K, L, sig1, log)

    sig2 = 0.35 * np.sqrt(sig1 - np.mean((stage1 - noised_img)**2))
    if log:
        print('sig2 = ', sig2)

    img = stage1

    stage2 = _denoise_image(K, L, sig2, log)

    return stage2