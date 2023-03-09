"""Module noref metrics contains no reference image quality metrics."""
import numpy as np
from skimage import filters
from scipy.stats import entropy
import pywt


def range_norm(vol):
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
    return out

def image_sharpness(vol):
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
    norm_vol = range_norm(vol)
    edges = filters.sobel(norm_vol)
    shrp_score = np.sum(edges)/vol.size
    return shrp_score

def shannon_entropy(vol):
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
    norm_vol = range_norm(vol)
    bins = 256
    [hist, _] = np.histogram(norm_vol, bins)
    ent = entropy(hist)
    return ent

def wavelet_average(vol):
    """Computes the Average wavelet energy of a volume
    Parameters
    ----------
    vol : numpy.ndarray
        Volume to calculate wavelet energy of
    Returns
    -------
    energy_avg : double
        Average wavelet energy of volume
    """
    norm_vol = range_norm(vol)
    coeffs = pywt.dwt2(norm_vol, 'db1')
    image_l = np.sqrt(coeffs[0]**2)
    image_h = np.sqrt(coeffs[1][0]**2 + coeffs[1][1]**2 + coeffs[1][2]**2)
    energy_l = np.sum(image_l**2/image_l.size)
    energy_h = np.sum(image_h**2/image_h.size)
    energy_avg = (energy_l + energy_h)/2
    return energy_avg
