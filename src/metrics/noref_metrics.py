import numpy as np
from skimage import filters
from scipy.stats import entropy
from skimage.transform import resize
import multiprocessing as mp
import pywt

import pandas as pd
import scipy.io as sio
import glob, os
import time
#from volshow import vol_show as volshow

def rangeNorm(vol):
	"""Normalizes the intensity range of a volume from 0.0 to 1.0
    Parameters
    ----------
    vol : numpy.ndarray
    	Volume to normalize as numpy array
    Returns
    -------
    out : numpy.ndarray
        Normalized volume
    """
	inmax = np.amax(vol)
	inmin = np.amin(vol)
	out = (vol-inmin)/(inmax-inmin)
	return(out)

def imageSharpness(vol):
	"""Computes the sharpness of a volume
	Parameters
	----------
	vol : numpy.ndarray
		Volume to calculate sharpness of
	Returns
	-------
	shrp_score : double
		Normalized sharpness score
	"""
	norm_vol = rangeNorm(vol)
	edges = filters.sobel(norm_vol)
	shrp_score = np.sum(edges)/vol.size
	return(shrp_score)

def shannonEntropy(vol):
	"""Computes the Shannon entropy of a volume
	Parameters
	----------
	vol : numpy.ndarray
		Volume to calculate entropy of
	Returns
	-------
	ent : double
		Shannon entropy of volume
	"""
	norm_vol = rangeNorm(vol)
	bins = 256
	[hist, rnge] = np.histogram(norm_vol, bins)
	ent = entropy(hist)
	return(ent)

def waveletAverage(vol):
	"""Computes the Average wavelet energy of a volume
	Parameters
	----------
	vol : numpy.ndarray
		Volume to calculate wavelet energy of
	Returns
	-------
	e_Avg : double
		Average wavelet energy of volume
	"""
	norm_vol = rangeNorm(vol)
	coeffs = pywt.dwt2(norm_vol, 'db1')
	I_L = np.sqrt(coeffs[0]**2)
	I_H = np.sqrt(coeffs[1][0]**2 + coeffs[1][1]**2 + coeffs[1][2]**2)
	e_L = np.sum(I_L**2/I_L.size)
	e_H = np.sum(I_H**2/I_H.size)
	e_Avg = (e_L + e_H)/2
	return(e_Avg)



# e_Avg = waveletAverage(srvol)
# print(e_Avg)

# ent = shannonEntropy(srvol)
# ent2 = shannonEntropy(normvol)
# print(ent)
# print(ent2)

# thing = imageSharpness(srvol)
# thing2 = imageSharpness(normvol)
# print(thing)
# print(thing2)