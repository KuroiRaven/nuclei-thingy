import matplotlib.pylab as plt
import numpy as np


def myfmad(array, axis=0, sigma_conv=True):
    """"""
    if axis == 0:
        step = abs(array-np.nanmedian(array, axis=axis))
    else:
        step = abs(array-np.nanmedian(array, axis=axis)[:, np.newaxis])
    return np.nanmedian(step, axis=axis)*[1, 1.48][int(sigma_conv)]
# kind = inter => return_border=true <== mask de toutes les valeurs mais on veut la valeur index 2


def rm_outliers(array, m=1.5, kind='sigma', axis=0, return_borders=False):
    if type(array) != np.ndarray:
        array = np.array(array)

    if m != 0:
        array[array == np.inf] = np.nan
        #array[array!=array] = np.nan

        if kind == 'inter':
            interquartile = np.nanpercentile(
                array, 75, axis=axis) - np.nanpercentile(array, 25, axis=axis)
            inf = np.nanpercentile(array, 25, axis=axis)-m*interquartile
            sup = np.nanpercentile(array, 75, axis=axis)+m*interquartile
            mask = (array >= inf) & (array <= sup)
        if kind == 'sigma':
            sup = np.nanmean(array, axis=axis) + m * \
                np.nanstd(array, axis=axis)
            inf = np.nanmean(array, axis=axis) - m * \
                np.nanstd(array, axis=axis)
            mask = abs(array-np.nanmean(array, axis=axis)) <= m * \
                np.nanstd(array, axis=axis)
        if kind == 'mad':
            median = np.nanmedian(array, axis=axis)
            mad = np.nanmedian(abs(array-median), axis=axis)
            sup = median+m * mad * 1.48
            inf = median-m * mad * 1.48
            mask = abs(array-median) <= m * mad * 1.48
    else:
        mask = np.ones(len(array)).astype('bool')

    if return_borders:
        return mask,  array[mask], sup, inf
    else:
        return mask,  array[mask]
