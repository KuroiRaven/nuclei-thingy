import getopt
import glob as glob

import matplotlib
import matplotlib.pylab as plt
import numpy as np
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from tqdm import tqdm
from models.analysis import Analysis
from services.utils import myfmad, rm_outliers

matplotlib.use('Qt5Agg')

pathFile = '../Bursting trial_34.czi'

analyzis = Analysis(pathFile)

time_index = 10

cutoff_sphere_z = 5  # fixed with the histogram distribution // Intensité du noyau
cutoff_spot_z = 40  # fixed with the histogram distribution // Intensité des spots
tresh_volume = 500  # minimum element for a nucleus to be detected as a nucleus
tresh_dist = np.sqrt(3)  # minimum distance for the nearest neighbor to be valid
k_cluster = 80  # amount of nucleus we're looking for
radius_cell_pxl = 7.5  # radius of a nucleus in pxl

frame = analyzis.frames[time_index]
voxel = frame.voxel
voxelNormalized = frame.getNormalizedVoxel()

#x,y = np.histogram(np.ravel(voxelNormalized[40]),bins=100)
# plt.plot(x,y)

tresh = voxelNormalized > cutoff_sphere_z

locations = []
for zSlice in range(len(voxelNormalized)):
    # all coordinates that are preserved after masking
    x, y = np.where(tresh[zSlice])
    tab = np.array([x, y, zSlice*np.ones(len(x))]).T
    locations.append(tab)
locations = np.vstack(locations)

# Recenter the axes + Z axes converted to pixel
locations[:, 0] -= analyzis.imageData.sizes.x/2
locations[:, 1] -= analyzis.imageData.sizes.y/2
locations[:, 2] -= analyzis.imageData.sizes.z/2
locations[:, 2] *= analyzis.imageData.slicePixelRatio

# Removes all noises
borders = np.array_split(np.arange(len(locations)), 5000)
all_min_dist = []
for index in tqdm(borders):
    sub_table = locations[index]
    dist = (sub_table[:, 0]-locations[:, 0][:, np.newaxis])**2+(sub_table[:, 1] -
                                                                locations[:, 1][:, np.newaxis])**2+(sub_table[:, 2]-locations[:, 2][:, np.newaxis])**2
    dist = np.where(dist == 0, np.nan, dist)
    dist = np.nanmin(dist, axis=0)
    all_min_dist.append(dist)
all_min_dist = np.array(all_min_dist)
all_min_dist = np.sqrt(np.hstack(all_min_dist))
# actual noise removal
locations_cells = locations[all_min_dist <= tresh_dist]

# creates a DF that contains the coordinates with names for each dimension
df = pd.DataFrame(locations_cells, columns=['x', 'y', 'z'])

# define center for each cluster
kmeans = KMeans(n_clusters=k_cluster, init='k-means++', random_state=0).fit(df)
centers_fitted = kmeans.cluster_centers_
labels_fitted = kmeans.labels_

# Dipole removal
standard_cluster = []
radius_cluster = []
for zSlice in np.unique(labels_fitted):
    pos_cluster = locations_cells[labels_fitted == zSlice]
    cluster_center = kmeans.cluster_centers_[zSlice]
    std_pts = np.sum((pos_cluster-cluster_center)**2, axis=1)
    radius_cluster.append(np.median(np.sqrt(std_pts)))
    standard_cluster.append(np.sqrt(np.sum(std_pts))/len(pos_cluster))
standard_cluster = np.array(standard_cluster)
radius_cluster = np.array(radius_cluster)

# counts the number of points that are contained in a nucleus
nb_points = np.array([np.sum(labels_fitted == i)
                     for i in np.sort(np.unique(labels_fitted))])

# Keeps all the points that revolve to a cluster that has more than
mask1 = (nb_points > tresh_volume)
mask2 = rm_outliers(standard_cluster)[0]
mask = mask1 & mask2
labels_kept = np.arange(len(nb_points))[mask]

mask_cluster = np.in1d(labels_fitted, labels_kept)
centers_fitted_kept = centers_fitted[mask]  # removes falsly found centers

# distance between cluster centers (euclidian distance => Pythagore)
dist_cluster = np.sqrt((centers_fitted_kept[:, 0]-centers_fitted_kept[:, 0][:, np.newaxis])**2+(centers_fitted_kept[:, 1] -
                       centers_fitted_kept[:, 1][:, np.newaxis])**2+(centers_fitted_kept[:, 2]-centers_fitted_kept[:, 2][:, np.newaxis])**2)
tresh_dist_cluster = dist_cluster < (radius_cell_pxl*2)
rows, cols = np.where(tresh_dist_cluster)
cluster_twin = (rows > cols)  # keeps the upper triangle of the symetric matrix
rows = rows[cluster_twin]
cols = cols[cluster_twin]

# replace label of the falsly divided nuclei in order to merge them
for r, c in zip(rows, cols):
    labels_fitted[labels_fitted == labels_kept[r]] = labels_kept[c]

# removes the deleted labels by taking the unique values of labels_fitted
final_labels = np.unique(labels_fitted[mask_cluster])
print('[INFO] Number of cells detected : %.0f' % (len(final_labels)))

# finds new center for each cluster
centers = []
for clusterLabel in final_labels:
    centers.append(
        np.mean(locations_cells[labels_fitted == clusterLabel], axis=0))
centers = np.array(centers)

# Gets back to image referential
centers_bulle = centers.copy()
centers_bulle[:, 2] /= analyzis.imageData.slicePixelRatio
centers_bulle[:, 0] += analyzis.imageData.sizes.x/2
centers_bulle[:, 1] += analyzis.imageData.sizes.y/2
centers_bulle[:, 2] += analyzis.imageData.sizes.z/2

# Every coordinates of the 3D "cube" representation of the image
x = np.arange(analyzis.imageData.sizes.x)*np.ones(analyzis.imageData.sizes.y)[:, np.newaxis] * \
    np.ones(analyzis.imageData.sizes.z)[:, np.newaxis][:, :, np.newaxis]
y = np.ones(analyzis.imageData.sizes.x)*np.arange(analyzis.imageData.sizes.y)[:, np.newaxis] * \
    np.ones(analyzis.imageData.sizes.z)[:, np.newaxis][:, :, np.newaxis]
z = np.ones(analyzis.imageData.sizes.x)*np.ones(analyzis.imageData.sizes.y)[:, np.newaxis] * \
    np.arange(analyzis.imageData.sizes.z)[:, np.newaxis][:, :, np.newaxis]

voxel_map_bulle = np.zeros(np.shape(voxelNormalized)).astype(bool)
for bulle in tqdm(centers_bulle):
    # distance from each center > radius
    mask_bulle = np.sqrt((x-bulle[0])**2+(y-bulle[1])**2 +
                         ((z-bulle[2])*analyzis.imageData.slicePixelRatio)**2) < (1.5*radius_cell_pxl)  # redefine radius_cell to have something that fits better
    voxel_map_bulle = voxel_map_bulle | mask_bulle
voxel_map_bulle = np.transpose(voxel_map_bulle, (0, 2, 1))  # Rotate matrix

# x1, y1 = np.histogram(np.ravel(voxelNormalized[voxel_map_bulle]), bins=100)
# plt.plot(y1[1:], np.log10(x1))

slices, height, width = voxel_map_bulle.shape
spotZones = []
for slice in range(slices):
    spotZones.append(voxelNormalized[slice][voxel_map_bulle[slice]])
print(list(spotZones))

voxel_spots = (voxelNormalized > cutoff_spot_z) & (voxel_map_bulle)
z, x, y = np.where(voxel_spots)

spots_locations = (np.array([x, y, z]).T).astype('float')

# recenter coordinates for plot
spots_locations[:, 0] -= analyzis.imageData.sizes.x/2
spots_locations[:, 1] -= analyzis.imageData.sizes.y/2
spots_locations[:, 2] -= analyzis.imageData.sizes.z/2
spots_locations[:, 2] *= analyzis.imageData.slicePixelRatio

# Plotting
ax = plt.figure().add_subplot(projection='3d')
ax.scatter(
    locations_cells[mask_cluster][::10, 2],
    locations_cells[mask_cluster][::10, 1],
    locations_cells[mask_cluster][::10, 0],
    c=labels_fitted[mask_cluster][::10],
    s=3, cmap='tab20', alpha=0.15)
# ax.scatter(centers_fitted_kept[:, 2], centers_fitted_kept[:,
#          1], centers_fitted_kept[:, 0], color='k', alpha=1)
#ax.scatter(centers[:, 2], centers[:, 1], centers[:, 0], color='b', alpha=1)
ax.scatter(spots_locations[:, 2], spots_locations[:, 1],
           spots_locations[:, 0], color='r', alpha=1)
plt.xlim(-256, 256)
plt.show()

for slice in range(70):
    #slice = 39
    plt.figure()
    plt.imshow(voxelNormalized[slice])
    plt.imshow(voxel_map_bulle[slice], alpha=0.3, cmap='Reds_r')
    plt.scatter(y[z == slice], x[z == slice], color='r', s=1, alpha=0.1)
