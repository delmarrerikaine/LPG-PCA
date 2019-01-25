# https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.medfilt.html

import scipy.signal as sig

def denoise(image, noise_std_dev):
    return sig.medfilt(image)