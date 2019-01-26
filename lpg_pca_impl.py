import numpy as np
from multiprocessing import Pool
import os
from sklearn.feature_extraction import image


cores_count = os.cpu_count() - 1 # don't use all cores, your UI may start to lag

def _denoise_pixel(img, x, y, K, L, sig):
    def getBlock(x, y):
        return img[x - halfK: x + halfK + 1, y - halfK: y + halfK + 1]

    # def mse(block):
    #     return np.mean((block - target)**2)
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
    rng = halfL - halfK
    blocks = image.extract_patches_2d(
        img[max(K, x - rng) - halfK : min(x + rng + 1, dim2 - K) + halfK,
        max(K, y - rng) - halfK : min(y + rng + 1, dim1 - K) + halfK], (K, K))
    
    # Sort by MSE
    sortIndexes = ((blocks - target)**2).reshape(blocks.shape[0], m, order = 'F').mean(axis = 1).argsort()

    # Construct the training matrix with the target and the best blocks reshaped into columns.
    trainingMatrix = blocks[sortIndexes].reshape(blocks.shape[0], m, order = 'F').swapaxes(1, 0)[:,:n+1]

    mean = trainingMatrix.mean(axis=1)
    trainingMatrix = trainingMatrix - mean.reshape(m, 1)
    noiseCov = sig**2 * np.eye(m, m)
    inputCov = (trainingMatrix @ trainingMatrix.T)/n
    eigvectors = np.linalg.eig(inputCov)[1]
    PX = eigvectors.T

    transInput = PX @ trainingMatrix

    transNoiseCov = PX @ noiseCov @ PX.T
    transInputCov = (transInput @ transInput.T)/n
    transDenoisedOutCov = np.maximum(np.zeros(transInputCov.shape), transInputCov - transNoiseCov)

    shrinkCoef = np.diag(transDenoisedOutCov)/(np.diag(transDenoisedOutCov) + np.diag(transInputCov))
    Y1 = transInput[:, 0] * shrinkCoef
    X1 = PX.T @ Y1 + mean
    return X1[m//2]


def _denoise_patch(img, left_x, right_x, left_y, right_y, K, L, sig, log):
    return [(x, left_y, right_y, [_denoise_pixel(img, x, y, K, L, sig) for y in range(left_y, right_y)]) 
            for x in range(left_x, right_x)]


def _denoise_image(img, K, L, sig, log):
    global outImg

    outImg = np.copy(img)
    width, height = img.shape
    halfL = L // 2
    halfK = K // 2

    def denoisePatchCallback(result): # [x, left_y, right_y, values]
        global outImg

        for elem in result:
            x, y_left, y_right, data = elem
            outImg[x, y_left:y_right] = data

    global pool
    global cores_count

    # parallel
    partitions = np.linspace(halfK, width - halfK, num=cores_count + 1)
    progress = np.array([])

    for i, partition in enumerate(partitions):
        if(i == cores_count):
            continue

        left_x = int(partitions[i])
        right_x = int(partitions[i + 1])
        left_y = halfK
        right_y = height - halfK

        pool.apply_async(_denoise_patch, (img, left_x, right_x, left_y, right_y, K, L, sig, log,), callback=denoisePatchCallback)

    pool.close()
    pool.join()

    # non-parallel:
    # for x in range(halfK, width - halfK):
    #     if log:
    #         print(x)
    #     for y in range(halfK, height - halfK):
    #         outImg[x, y] = _denoise_pixel(img, x, y, K, L, sig)

    return outImg


def denoise(noised_img, sig1, K=3, L=21, log=False):
    global pool
    global cores_count

    pool = Pool(cores_count)

    stage1 = _denoise_image(noised_img, K, L, sig1, log)

    sig2 = 0.35 * np.sqrt(sig1 - np.mean((stage1 - noised_img)**2))
    if log:
        print('sig2 = ', sig2)

    pool = Pool(cores_count)

    stage2 = _denoise_image(stage1, K, L, sig2, log)

    # pool.join()
    pool.terminate()

    return stage2
