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

def match_nearest_ndim(array1, array2, max_dist=None, znorm=False):
    """return a table [idx1,idx2,num1,num2,distance] matching the closest element from two arrays. Remark : algorithm very slow by conception if the arrays are too large."""
    
    if type(array1)!=np.ndarray:
        array1 = np.array(array1)
    if type(array2)!=np.ndarray:
        array2 = np.array(array2)    
    if not (np.product(~np.isnan(array1))*np.product(~np.isnan(array2))):
        print('there is a nan value in your list, remove it first to be sure of the algorithme reliability')
    
    if len(np.shape(array1))<2:
        array1 = array1[:,np.newaxis].T

    if len(np.shape(array2))<2:
        array2 = array2[:,np.newaxis].T

    len1 = len(array1.T)
    len2 = len(array2.T)
    
    mask_na1 = (1-np.sum(np.isnan(array1),axis=0)).astype('bool')
    mask_na2 = (1-np.sum(np.isnan(array2),axis=0)).astype('bool')
    
    index1 = np.arange(len1)[mask_na1] ; index2 = np.arange(len2)[mask_na2]  
    array1 = array1[:,mask_na1] ;  array2 = array2[:,mask_na2]
    liste1 = np.arange(len1)[:,np.newaxis]*np.hstack([np.ones(len1)[:,np.newaxis],np.zeros(len1)[:,np.newaxis]])
    liste2 = np.arange(len2)[:,np.newaxis]*np.hstack([np.ones(len2)[:,np.newaxis],np.zeros(len2)[:,np.newaxis]])
    liste1 = liste1.astype('int') ; liste2 = liste2.astype('int')
    
    if znorm:
        med_vec = np.median(np.hstack([array1,array2]),axis=1)[:,np.newaxis]
        mad_vec = myfmad(np.hstack([array1,array2]),axis=1)[:,np.newaxis]
    else:
        med_vec = 0
        mad_vec = 1
        
    array1 = array1 - med_vec
    array2 = array2 - med_vec
    
    array1 = array1/mad_vec
    array2 = array2/mad_vec
    
    #ensure that the probability for two close value to be the same is null
    if len(array1.T)>1:
        dmin = np.min(np.diff(np.sort(array1,axis=-1),axis=-1),axis=-1)
    else:
        dmin = np.zeros(len(array1))
    if len(array2.T)>1:
        dmin2 = np.min(np.diff(np.sort(array2,axis=-1),axis=-1),axis=-1)
    else:
        dmin2 = np.zeros(len(array2))
        
    array1_r = array1 + 0.001*np.random.randn(len(array1.T))*dmin[:,np.newaxis]
    array2_r = array2 + 0.001*np.random.randn(len(array2.T))*dmin2[:,np.newaxis]
    #match nearest
    m = np.sum([abs(array2_r[j]-array1_r[j][:,np.newaxis]) for j in np.arange(len(array1_r))],axis=0)
    arg1 = np.argmin(m,axis=0)
    arg2 = np.argmin(m,axis=1)
    mask = (np.arange(len(arg1)) == arg2[arg1])
    liste_idx1 = arg1[mask]
    liste_idx2 = arg2[arg1[mask]]
    array1_k = array1[:,liste_idx1]
    array2_k = array2[:,liste_idx2]
    
    array1_k = array1_k*mad_vec
    array2_k = array1_k*mad_vec

    array1_k = array1_k+med_vec
    array2_k = array1_2+med_vec    

    liste_idx1 = index1[liste_idx1]
    liste_idx2 = index2[liste_idx2]    
 
    mat = np.hstack([liste_idx1[:,np.newaxis],liste_idx2[:,np.newaxis],
                      array1_k.T,array2_k.T,(array1_k-array2_k).T]) 
    
    if max_dist is not None:
       mat = mat[(abs(mat[:,-1])<max_dist)]
    
    return mat  
  