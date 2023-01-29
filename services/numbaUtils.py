from numba import prange, njit
from numpy import sqrt, float32 as npFloat32, float64 as npFloat64, zeros, inf, where, sum, array, median, shape, ones, bool8, ndarray, empty
from .utils import rmOutliersInterWithBorders, filter3D


@njit(parallel=True, nogil=True)
def getDistances(locations):
    numberLocations = locations.shape[0]
    allMinDists = zeros(numberLocations, dtype=npFloat32)
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
        spotid = onesSpots * (nucleusId + 1 - penality)
        spotFrame = onesSpots * frameId
        parentNucleus = onesSpots * nucleusId
        return (array([spoti, spotX, spotY, spotZ, spotid, parentNucleus, spotFrame]).T, 0)
    else:
        return (None, 1)


@njit(parallel=True, nogil=True)
def getSpots(nucleiCenters: ndarray, positions: tuple[ndarray, ndarray, ndarray], voxelNormalized: ndarray, slicePixelRatio: float, radiusCellPixel: float, frameId: int):
    penality = 0
    numberNuclei = nucleiCenters.shape[0]
    thresholds = []
    allSpots = []
    voxelMaskNuclei = zeros(shape(voxelNormalized), dtype=bool8)
    for i in prange(numberNuclei):
        nucleus = nucleiCenters[i]
        maskNucleus: ndarray = sqrt((positions[0]-nucleus[1])**2+(positions[1]-nucleus[0])**2+((positions[2]-nucleus[2])*slicePixelRatio)**2) < (1.5*radiusCellPixel)
        voxelMaskNuclei = voxelMaskNuclei | maskNucleus
        print(maskNucleus)
        maskedVoxelNormalized = filter3D(voxelNormalized, maskNucleus)  # array([voxelNormalized[pixelLine][maskNucleus[pixelLine]] for pixelLine in range(voxelNormalized.shape[0])])
        # voxelNormalized[maskNucleus]
        threshold = rmOutliersInterWithBorders(maskedVoxelNormalized)[2]
        thresholds.append(threshold)
        spotsNucleus = getSpotsNucleus(i, threshold, maskNucleus, maskedVoxelNormalized, frameId, penality)
        if spotsNucleus[1] > 0:
            penality += spotsNucleus[1]
        else:
            allSpots.append(spotsNucleus[0])

    thresholdsAsArray = array(thresholds)

    return (allSpots, thresholdsAsArray, voxelMaskNuclei)
