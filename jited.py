""" from numba import config, jit, njit, vectorize, prange, threading_layer,  gdb_init, gdb_breakpoint, int64
from time import time
from models.voxel import Voxel
from numpy import ndarray, zeros, fromiter, array, fromiter, float64, where, array, ones, vstack, arange, intp, full
import numpy as np
from numpy.typing import NDArray
from typing import Any
import numba


""" @ njit(parallel=True)


def distance(location, locations):
    minDists = []
    for index in prange(locations.shape[0]):
        otherPoint = locations[index]
        dist = (location[0] - otherPoint[0])**2+(location[1] - otherPoint[1])**2+(location[2] - otherPoint[2])**2
        if dist > 0:
            minDists.append(dist)
    minDist = array(minDists).min()
    return minDist


startTime = time()
test = fromiter((distance(position, locations) for position in [locations[2]]), dtype='object_')
print(time() - startTime) """

""", parallel = True, nogil = True, debug = True  """


from numba import vectorize, float64, int32, int64, float32, njit, prange
from numpy import ndarray
@njit(parallel=True, nogil=True)
def parallelDistances(locations):
    numberLocations = locations.shape[0]
    allMinDists = zeros(numberLocations, dtype=np.float32)
    for i in prange(numberLocations):
        location = locations[i, :]
        currentMinDist = np.inf
        for idSecondLoc in range(numberLocations):
            if i == idSecondLoc:
                continue
            otherLocation = locations[idSecondLoc, :]
            dist = (location[0] - otherLocation[0])**2+(location[1] - otherLocation[1])**2+(location[2] - otherLocation[2])**2
            currentMinDist = min(currentMinDist, dist)
        allMinDists[i] = currentMinDist

    return allMinDists


# locationAsNPArray = locations
startTime = time()
myDistances = parallelDistances(locations)
print(time() - startTime)
####################################################################
####################################################################
####################################################################


# @numba.njit()
# def parallelDistances(positions):
#     test = 0
#     for i in numba.prange(positions):
#         test += i
#     return test


# parallelDistances(10000000000000)

@njit(parallel=True, nogil=True)
def parallel_distances(locations):
    n_loc = locations.shape[0]
    min_dist = np.zeros(n_loc, dtype=np.float32)

    for i in prange(n_loc):
        p1 = locations[i, :]
        min_dist_loc = np.inf
        for j in range(n_loc):
            if i == j:
                continue
            p2 = locations[j, :]
            dist = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)
            min_dist_loc = min(min_dist_loc, dist)
        min_dist[i] = min_dist_loc
    return min_dist


startTime = time()
distances = parallel_distances(locations)
print(time() - startTime)
 """


@njit(parallel=True, nogil=True)
def sumAxisZero(array):
    nbCols = array[0].shape[0]
    sums = np.empty(nbCols)
    for i in prange(nbCols):
        sums[i] = np.sum(array[:, i])
    return sums


@vectorize([int32(int32, int32),
            int64(int64, int64),
            float32(float32, float32),
            float64(float64, float64)], )
def sumAxisZeroVect(x, y):
    return x + y


testMatrix = np.random.rand(50000, 30000)
startNp = time()
testSumAxis0 = np.sum(testMatrix, axis=0)
print("np: " + str(time() - startNp))

startJit = time()
testSumAxisZero = sumAxisZero(testMatrix)
print("numba: " + str(time() - startJit))

startVect = time()
sumAxisZeroVect.reduce(testMatrix, axis=0)
print("vectorized:" + str(time() - startVect))


@njit()
def maskWithWhere():
    testMatrix = np.array([[
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
    ]])

    maskArrayC = np.array([[
        [False, False, True, False, True],
        [False, True, True, False, False],
    ]])

    masked3d = np.where((testMatrix >= 0.2) & maskArrayC)
    return masked3d
