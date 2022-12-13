from numpy import array as npArray
from numpy import nanmedian, ndarray, unique, sort, arange, sqrt, where, mean, in1d, newaxis
from services.utils import myfmad
from sklearn.cluster import KMeans
import pandas as pd
from time import time

from .nucleus import Nucleus
from services.settings import minClustervolume, minRadiusCell, approximateAmountOfNuclei
from services.numbaUtils import getDipoles
from services.utils import rm_outliers


class Frame(object):
    frameId: int
    voxel: ndarray
    nuclei: list[Nucleus]

    def __init__(self, frameId: int, voxel: ndarray):
        self.frameId = frameId
        self.voxel = voxel

    def getNormalizedVoxel(self) -> ndarray:
        return npArray(list(map(self.__normalizeSlice, self.voxel)))

    def __normalizeSlice(self, voxelSlice):
        inter = voxelSlice - nanmedian(voxelSlice)
        inter /= myfmad(inter)
        return inter

    def getNuclei(self, locations) -> ndarray:
        startTime = time()
        df = pd.DataFrame(locations, columns=['x', 'y', 'z'])

        # define center for each cluster
        tempClusterCenters, pixelClusterIds, uniqueLabels = self.__getClusterCenters(df)

        # Dipole removal
        timeBefDipole = time()
        standard_cluster, radius_cluster = getDipoles(tempClusterCenters, uniqueLabels, pixelClusterIds, locations)
        print("[INFO] getDipoles: " + str(time() - timeBefDipole))

        # counts the number of points that are contained in a nucleus
        nb_points = npArray([sum(pixelClusterIds == i) for i in sort(uniqueLabels)])

        # Keeps all the points that revolve to a cluster that has more than
        maskMinThreshold = (nb_points > minClustervolume)
        maskOutliers = rm_outliers(standard_cluster)[0]
        mask = maskMinThreshold & maskOutliers

        clusterCentersEnoughPoints = tempClusterCenters[mask]  # removes falsly found centers
        finalClusterIds = arange(len(nb_points))[mask]

        # distance between cluster centers (euclidian distance => Pythagore)
        clusterDistances = sqrt((clusterCentersEnoughPoints[:, 0]-clusterCentersEnoughPoints[:, 0][:, newaxis])**2+(clusterCentersEnoughPoints[:, 1] - clusterCentersEnoughPoints[:, 1][:, newaxis])**2+(clusterCentersEnoughPoints[:, 2]-clusterCentersEnoughPoints[:, 2][:, newaxis])**2)
        maskClusterDistances = clusterDistances < (minRadiusCell*2)
        rows, cols = where(maskClusterDistances)
        clusterTwin = (rows > cols)  # keeps the upper triangle of the symetric matrix
        rows = rows[clusterTwin]
        cols = cols[clusterTwin]

        # replace label of the falsly divided nuclei in order to merge them
        for r, c in zip(rows, cols):
            pixelClusterIds[pixelClusterIds == finalClusterIds[r]] = finalClusterIds[c]

        # removes the deleted labels by taking the unique values of labelsFitted
        maskClusters = in1d(pixelClusterIds, finalClusterIds)
        final_labels = unique(pixelClusterIds[maskClusters])
        print('[INFO] Number of cells detected : %.0f' % (len(final_labels)))
        print("[INFO] getNuclei: " + str(time() - startTime))

        # finds new center for each cluster
        centers = []
        for clusterLabel in final_labels:
            centers.append(mean(locations[pixelClusterIds == clusterLabel], axis=0))
        return npArray(centers)

    def __getClusterCenters(self, dataFrame):
        startTime = time()
        kmeans = KMeans(n_clusters=approximateAmountOfNuclei, init='k-means++', random_state=0).fit(dataFrame)
        clusterCenters: ndarray = kmeans.cluster_centers_
        pixelClusterIds: ndarray = kmeans.labels_  # correspondance à un cluster
        uniqueLabels = unique(pixelClusterIds)
        print("[INFO] GetClusterCenters: " + str(time() - startTime))
        return (clusterCenters, pixelClusterIds, uniqueLabels)
