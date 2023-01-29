import matplotlib.pylab as plt
import numpy as np
from numba import njit, prange, vectorize, int32, int64, float32, float64


@vectorize([int32(int32, int32),
            int64(int64, int64),
            float32(float32, float32),
            float64(float64, float64)], )
def sumAxisZero(x, y):
    return x + y


@njit
def np_apply_along_axis(func1d, axis, arr):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result


@njit()
def median(array, axis=0):
    return np_apply_along_axis(np.median, axis, array)


@njit()
def nanmedian(array, axis=0):
    return np_apply_along_axis(np.median, axis, array)


@njit()
def nanmean(array, axis=0):
    return np_apply_along_axis(np.mean, axis, array)


@njit()
def nanstd(array, axis=0):
    return np_apply_along_axis(np.nanstd, axis, array)


def myfmad(array, axis=0, sigma_conv=True):
    if axis == 0:
        step = np.abs(array-nanmedian(array, axis=axis))
    else:
        step = np.abs(array-nanmedian(array, axis=axis)[:, np.newaxis])
    return np.nanmedian(step, axis=axis)*[1, 1.48][int(sigma_conv)]
# kind = inter => return_border=true <== mask de toutes les valeurs mais on veut la valeur index 2


@njit()
def nan_percentile_axis0(arr, percentiles):
    """Faster implementation of np.nanpercentile

    This implementation always takes the percentile along axis 0.
    Uses numba to speed up the calculation by more than 7x.

    Function is equivalent to np.nanpercentile(arr, <percentiles>, axis=0)

    Params:
        arr (np.array): Array to calculate percentiles for
        percentiles (np.array): 1D array of percentiles to calculate

    Returns:
        (np.array) Array with first dimension corresponding to
            values as passed in percentiles

    """

    shape = arr.shape
    arr = arr.reshape((arr.shape[0], -1))
    out = np.empty((percentiles, arr.shape[1]))
    for i in range(arr.shape[1]):
        out[:, i] = np.nanpercentile(arr[:, i], percentiles)
    shape = (out.shape[0], *shape[1:])
    return out.reshape(shape)


@njit(parallel=True, nogil=True)
def filter3D(array, mask):
    if (array.shape != mask.shape):
        raise Exception('Array and mask dimensions are different, please ensure that you chose the right ones')

    shape = array.shape
    slices, width, length = shape
    output = np.empty(shape)

    for i in prange(slices):
        for j in range(width):
            arrayOneD = array[i][j]
            maskOneD = mask[i][j]
            output[i][j] = maskArray1d(arrayOneD, maskOneD)

    return output


@njit(nogil=True)
def maskArray1d(array, mask):
    return array[mask]


@njit()
def rmOutliersInter(array: np.ndarray, m=1.5):
    if m == 0:
        raise Exception('m can\'t be equal to 0.')

    interquartile = nan_percentile_axis0(array, 75) - nan_percentile_axis0(array, 25)
    inf = nan_percentile_axis0(array, 25)-m*interquartile
    sup = nan_percentile_axis0(array, 75)+m*interquartile
    mask = (array >= inf) & (array <= sup)
    maskedArray = array[mask]

    return (mask,  maskedArray)


@njit()
def rmOutliersSigma(array: np.ndarray, m=1.5, axis=0):
    if m == 0:
        raise Exception('m can\'t be equal to 0.')

    if (array.ndim == 2):
        mask = np.abs(array-nanmean(array, axis=axis)) <= m * nanstd(array, axis=axis)
    else:
        mask = np.abs(array-np.nanmean(array)) <= m * np.nanstd(array)

    maskedArray = array[mask]

    return (mask,  maskedArray)


@njit()
def rmOutliersMad(array: np.ndarray, m=1.5, axis=0):
    if m == 0:
        raise Exception('m can\'t be equal to 0.')

    mask = np.abs(array-nanmean(array, axis=axis)) <= m * nanstd(array, axis=axis)

    if (array.ndim == 2):
        median = nanmedian(array, axis=axis)
        mad = nanmedian(np.abs(array-median))
    else:
        median = np.nanmedian(array)
        mad = np.nanmedian(np.abs(array-median))

    mask = np.abs(array-median) <= m * mad * 1.48
    maskedArray = array[mask]

    return (mask,  maskedArray)


@njit()
def rmOutliersInterWithBorders(array: np.ndarray, m=1.5):
    if m == 0:
        raise Exception('m can\'t be equal to 0.')

    interquartile = nan_percentile_axis0(array, 75) - nan_percentile_axis0(array, 25)
    inf = nan_percentile_axis0(array, 25)-m*interquartile
    sup = nan_percentile_axis0(array, 75)+m*interquartile
    mask = (array >= inf) & (array <= sup)
    maskedArray = array[mask] if array.ndim == 1 else filter3D(array, mask)

    return (mask,  maskedArray, sup, inf)


@njit()
def rmOutliersSigmaWithBorders(array: np.ndarray, m=1.5, axis=0):
    if m == 0:
        raise Exception('m can\'t be equal to 0.')

    if (array.ndim == 2):
        sup = nanmean(array, axis=axis) + m * nanstd(array, axis=axis)
        inf = nanmean(array, axis=axis) - m * nanstd(array, axis=axis)
        mask = np.abs(array-nanmean(array, axis=axis)) <= m * nanstd(array, axis=axis)
    else:
        sup = np.nanmean(array) + m * np.nanstd(array)
        inf = np.nanmean(array) - m * np.nanstd(array)
        mask = np.abs(array-np.nanmean(array)) <= m * np.nanstd(array)

    maskedArray = array[mask]

    return (mask,  maskedArray, sup, inf)


@njit()
def rmOutliersMadWithBorders(array: np.ndarray, m=1.5, axis=0):
    if m == 0:
        raise Exception('m can\'t be equal to 0.')

    mask = np.abs(array-nanmean(array, axis=axis)) <= m * nanstd(array, axis=axis)

    if (array.ndim == 2):
        median = nanmedian(array, axis=axis)
        mad = nanmedian(np.abs(array-median))
    else:
        median = np.nanmedian(array)
        mad = np.nanmedian(np.abs(array-median))

    sup = median+m * mad * 1.48
    inf = median-m * mad * 1.48
    mask = np.abs(array-median) <= m * mad * 1.48
    maskedArray = array[mask]

    return (mask,  maskedArray, sup, inf)


# def rm_outliers(array: np.ndarray, m=1.5, kind='sigma', axis=0, return_borders=False) -> tuple[np.ndarray, np.ndarray, np.ndarray | np.floating | None, np.ndarray | np.floating | None]:


def match_nearest_ndim(array1, array2, max_dist=None, znorm=False):
    """return a table [idx1,idx2,num1,num2,distance] matching the closest element from two arrays. Remark : algorithm very slow by conception if the arrays are too large."""

    if type(array1) != np.ndarray:
        array1 = np.array(array1)
    if type(array2) != np.ndarray:
        array2 = np.array(array2)
    if not (np.product(~np.isnan(array1))*np.product(~np.isnan(array2))):
        print('there is a nan value in your list, remove it first to be sure of the algorithme reliability')

    if len(np.shape(array1)) < 2:
        array1 = array1[:, np.newaxis].T

    if len(np.shape(array2)) < 2:
        array2 = array2[:, np.newaxis].T

    len1 = len(array1.T)
    len2 = len(array2.T)

    mask_na1 = (1-sumAxisZero.reduce(np.isnan(array1), axis=0)).astype('bool')
    mask_na2 = (1-sumAxisZero.reduce(np.isnan(array2), axis=0)).astype('bool')

    index1 = np.arange(len1)[mask_na1]
    index2 = np.arange(len2)[mask_na2]
    array1 = array1[:, mask_na1]
    array2 = array2[:, mask_na2]
    liste1 = np.arange(len1)[:, np.newaxis]*np.hstack([np.ones(len1)[:, np.newaxis], np.zeros(len1)[:, np.newaxis]])
    liste2 = np.arange(len2)[:, np.newaxis]*np.hstack([np.ones(len2)[:, np.newaxis], np.zeros(len2)[:, np.newaxis]])
    liste1 = liste1.astype('int')
    liste2 = liste2.astype('int')

    if znorm:
        med_vec = nanmedian(np.hstack([array1, array2]), axis=1)[:, np.newaxis]
        mad_vec = myfmad(np.hstack([array1, array2]), axis=1)[:, np.newaxis]
    else:
        med_vec = 0
        mad_vec = 1

    array1 = array1 - med_vec
    array2 = array2 - med_vec

    array1 = array1/mad_vec
    array2 = array2/mad_vec

    # ensure that the probability for two close value to be the same is null
    if len(array1.T) > 1:
        dmin = np.min(np.diff(np.sort(array1, axis=-1), axis=-1), axis=-1)
    else:
        dmin = np.zeros(len(array1))
    if len(array2.T) > 1:
        dmin2 = np.min(np.diff(np.sort(array2, axis=-1), axis=-1), axis=-1)
    else:
        dmin2 = np.zeros(len(array2))

    array1_r = array1 + 0.001*np.random.randn(len(array1.T))*dmin[:, np.newaxis]
    array2_r = array2 + 0.001*np.random.randn(len(array2.T))*dmin2[:, np.newaxis]
    # match nearest
    m = sumAxisZero.reduce([np.abs(array2_r[j]-array1_r[j][:, np.newaxis]) for j in np.arange(len(array1_r))], axis=0)
    arg1 = np.argmin(m, axis=0)
    arg2 = np.argmin(m, axis=1)
    mask = (np.arange(len(arg1)) == arg2[arg1])
    liste_idx1 = arg1[mask]
    liste_idx2 = arg2[arg1[mask]]
    array1_k = array1[:, liste_idx1]
    array2_k = array2[:, liste_idx2]

    array1_k = array1_k*mad_vec
    array2_k = array1_k*mad_vec

    array1_k = array1_k+med_vec
    array2_k = array2_k+med_vec

    liste_idx1 = index1[liste_idx1]
    liste_idx2 = index2[liste_idx2]

    mat = np.hstack([liste_idx1[:, np.newaxis], liste_idx2[:, np.newaxis],
                     array1_k.T, array2_k.T, (array1_k-array2_k).T])

    if max_dist is not None:
        mat = mat[(np.abs(mat[:, -1]) < max_dist)]

    return mat
