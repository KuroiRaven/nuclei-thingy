from numba import prange, njit
from numpy import sqrt, float32, zeros, inf, where, sum, array, median, shape, ones
from .utils import rm_outliers


@njit(parallel=True, nogil=True)
def getDistances(locations):
    numberLocations = locations.shape[0]
    allMinDists = zeros(numberLocations, dtype=float32)
    for i in prange(numberLocations):
        location = locations[i, :]
        currentMinDist = inf
        for idSecondLoc in range(numberLocations):
            if i == idSecondLoc:
                continue
            otherLocation = locations[idSecondLoc, :]
            dist = (location[0] - otherLocation[0])**2+(location[1] - otherLocation[1])**2+(location[2] - otherLocation[2])**2
            currentMinDist = min(currentMinDist, dist)
        allMinDists[i] = currentMinDist

    return sqrt(allMinDists)


@njit(parallel=True, nogil=True)
def getDipoles(clusterCenters, uniqueLabels, labelsFitted, locationsCells):
    standardCluster = []
    radiusCluster = []
    for labelId in prange(len(uniqueLabels)):
        zSlice = uniqueLabels[labelId]
        posCluster = locationsCells[labelsFitted == zSlice]
        stdPts = sum((posCluster-clusterCenters)**2, axis=1)
        radiusCluster.append(median(sqrt(stdPts)))
        standardCluster.append(sqrt(sum(stdPts))/len(posCluster))
    standardCluster = array(standardCluster)
    radiusCluster = array(radiusCluster)
    return (standardCluster, radiusCluster)


@njit(parallel=True, nogil=True)
def getSpotsNucleus(nucleusId, threshold, maskNucleus, voxelNormalized, frameId, penality):
    spotZ, spotX, spotY = where((voxelNormalized >= threshold) & (maskNucleus))
    spoti = array([voxelNormalized[zi, xi, yi] for xi, yi, zi in zip(spotX, spotY, spotZ)])
    onesSpots = ones(len(spotZ))
    # print('[INFO] Treshold set to I_lim = %.2f for spot in nucleus %.0f. Spots nb = %.0f' % (val_sup, counter+1-penality, len(spotx)))
    if len(spotZ):
        spotid = onesSpots * (nucleusId+1-penality)
        spotFrame = onesSpots * frameId
        parentNucleus = onesSpots * nucleusId
        return (array([spoti, spotX, spotY, spotZ, spotid, parentNucleus, spotFrame]).T, 0)
    else:
        return (None, 1)


@njit(parallel=True, nogil=True)
def getSpots(nucleiCenters, positions, voxelNormalized, slicePixelRatio, radiusCellPixel, frameId):
    penality = 0
    numberNuclei = nucleiCenters.shape[0]
    thresholds = zeros(numberNuclei, dtype=float32)
    allSpots = []
    voxelMaskNuclei = zeros(shape(voxelNormalized)).astype(bool)
    for i in prange(numberNuclei):
        nucleus = nucleiCenters[i]
        maskNucleus = sqrt((positions.x-nucleus[1])**2+(positions.y-nucleus[0])**2+((positions.z-nucleus[2])*slicePixelRatio)**2) < (1.5*radiusCellPixel)
        voxelMaskNuclei = voxelMaskNuclei | maskNucleus
        threshold = rm_outliers(voxelNormalized[maskNucleus], kind='inter', return_borders=True)[2]
        thresholds[i] = threshold
        spotsNucleus = getSpotsNucleus(i, threshold, maskNucleus, voxelNormalized, frameId, penality)
        if spotsNucleus[1] > 0:
            penality += spotsNucleus[1]
        else:
            allSpots.append(spotsNucleus[0])
    return (allSpots, thresholds, voxelMaskNuclei)
