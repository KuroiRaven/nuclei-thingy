from services.numbaUtils import getDistances, getSpots
from services.utils import match_nearest_ndim
from models.returnValueThread import ReturnValueThread
from models.analysis import Analysis
from tqdm import tqdm
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import numpy.typing as npt
import matplotlib.pylab as plt
import matplotlib
from time import time
import glob as glob

matplotlib.use('Qt5Agg')

pathFile = '../Bursting trial_34.czi'

startTime = time()

analyzis = Analysis(pathFile)

time_index = 10

cutoff_sphere_z = 5  # fixed with the histogram distribution // Intensité du noyau
tresh_volume = 500  # minimum element for a nucleus to be detected as a nucleus
tresh_dist = np.sqrt(3)  # minimum distance for the nearest neighbor to be valid
k_cluster = 80  # amount of nucleus we're looking for
radius_cell_pxl = 7.5  # radius of a nucleus in pxl
vicinity = 2  # proximity

time_indexes = [13]
for time_index in time_indexes:
    frame = analyzis.frames[time_index]
    voxel = frame.voxel
    voxelNormalized = frame.getNormalizedVoxel()

    # x,y = np.histogram(np.ravel(voxelNormalized[40]),bins=100)
    # plt.plot(x,y)

    tresh = voxelNormalized > cutoff_sphere_z

    def getLocations(slideId: int):
        x, y = np.where(tresh[slideId])  # all coordinates that are preserved after masking
        tab = np.array([x, y, slideId*np.ones(len(x))]).T
        return tab

    locationsTuple = tuple(map(getLocations, range(len(voxelNormalized))))
    locations = np.vstack(locationsTuple).astype(dtype=np.float64)

    recenterX = analyzis.imageData.sizes.x/2
    recenterY = analyzis.imageData.sizes.y/2
    recenterZ = analyzis.imageData.sizes.z/2
    # Recenter the axes + Z axes converted to pixel
    locations[:, 0] -= recenterX
    locations[:, 1] -= recenterY
    locations[:, 2] -= recenterZ
    locations[:, 2] *= analyzis.imageData.slicePixelRatio

    # Removes all noises
    allMinDists = getDistances(locations)

    # actual noise removal
    locations_cells = locations[allMinDists <= tresh_dist]

    nucleiCenters = frame.getNuclei(locations_cells)

    # Gets back to image referential
    nucleiCentersImagePosition = nucleiCenters.copy()
    nucleiCentersImagePosition[:, 2] /= analyzis.imageData.slicePixelRatio
    nucleiCentersImagePosition[:, 0] += analyzis.imageData.sizes.x/2
    nucleiCentersImagePosition[:, 1] += analyzis.imageData.sizes.y/2
    nucleiCentersImagePosition[:, 2] += analyzis.imageData.sizes.z/2

    # Every coordinates of the 3D "cube" representation of the image
    x = np.arange(analyzis.imageData.sizes.x)*np.ones(analyzis.imageData.sizes.y)[:, np.newaxis] * \
        np.ones(analyzis.imageData.sizes.z)[:, np.newaxis][:, :, np.newaxis]
    y = np.ones(analyzis.imageData.sizes.x)*np.arange(analyzis.imageData.sizes.y)[:, np.newaxis] * \
        np.ones(analyzis.imageData.sizes.z)[:, np.newaxis][:, :, np.newaxis]
    z = np.ones(analyzis.imageData.sizes.x)*np.ones(analyzis.imageData.sizes.y)[:, np.newaxis] * \
        np.arange(analyzis.imageData.sizes.z)[:, np.newaxis][:, :, np.newaxis]

    spot_table, all_tresh, voxel_map_bulle = getSpots(nucleiCentersImagePosition, (x, y, z), voxelNormalized, analyzis.imageData.slicePixelRatio, radius_cell_pxl, time_index)
    spot_table = np.vstack(spot_table).astype('int')
    spot_table = pd.DataFrame(spot_table, columns=['spot_i', 'spot_x', 'spot_y', 'spot_z', 'cell', 'cell_x', 'cell_y', 'cell_z', 'time'])

    # loc_max = []
    # for xi, yi, zi in zip(spot_table['spot_x'], spot_table['spot_y'], spot_table['spot_z']):
    #     cube = voxelNormalized[zi-vicinity:zi+vicinity+1, xi-vicinity:xi+vicinity+1, yi-vicinity:yi+vicinity+1]
    #     if len(cube) == 0:
    #         continue
    #     loc = np.where(cube == np.max(cube))
    #     crit = np.sum([abs(l-vicinity) for l in loc])
    #     loc_max.append(int(crit == 0))
    # spot_table['loc_max_qc'] = np.array(loc_max)
    # output = spot_table.loc[spot_table['loc_max_qc'] == 1]

    spot_table.to_csv('./Output_table_time_index_%s.csv' % (str(time_index).zfill(3)))

all_files = np.sort(glob.glob('./Output_table_time_index*csv'))

dataFrame = pd.concat([])

for i, f in enumerate(all_files):
    fileTemp = pd.read_csv(f, index_col=0)
    if not i:
        dataFrame = fileTemp.copy()
    else:
        dataFrame = pd.concat([dataFrame, fileTemp])

mapping = {}
c = -1
for timeIndex1, timeIndex2 in zip(time_indexes[0:-1], time_indexes[1:]):
    c += 1
    uniqueVals1 = dataFrame.loc[dataFrame['time'] == timeIndex1, ['cell_x', 'cell_y', 'cell_z', 'cell']]
    uniqueVals2 = dataFrame.loc[dataFrame['time'] == timeIndex2, ['cell_x', 'cell_y', 'cell_z', 'cell']]
    uniqueCellsByTimeFrame1 = np.array(uniqueVals1.drop_duplicates(subset=['cell'])[['cell_x', 'cell_y', 'cell_z']])
    uniqueCellsByTimeFrame2 = np.array(uniqueVals2.drop_duplicates(subset=['cell'])[['cell_x', 'cell_y', 'cell_z']])

    print(len(uniqueCellsByTimeFrame1), len(uniqueCellsByTimeFrame2))

    dist_x = uniqueCellsByTimeFrame1[:, 0]-uniqueCellsByTimeFrame2[:, 0][:, np.newaxis]
    dist_y = uniqueCellsByTimeFrame1[:, 1]-uniqueCellsByTimeFrame2[:, 1][:, np.newaxis]
    dist_z = uniqueCellsByTimeFrame1[:, 2]-uniqueCellsByTimeFrame2[:, 2][:, np.newaxis]

    dist = np.sqrt((dist_x)**2+(dist_y)**2+(dist_z)**2)

    index1 = np.argmin(dist, axis=0)
    index2 = np.argmin(dist, axis=1)

    match2_to_1 = index1[index2] - np.arange(len(index2))
    match1_to_2 = index2[index1] - np.arange(len(index1))

    if not c:
        name1 = np.arange(len(index1))
        mapping[timeIndex1] = np.array([np.arange(len(index1)), name1]).T

    name2 = np.zeros(len(index2)).astype('int')
    name2[match2_to_1 == 0] = mapping[timeIndex1][:, 1][index2[match2_to_1 == 0]]
    name2[match2_to_1 != 0] = np.max(name2)+np.arange(1, 1+sum(match2_to_1 != 0))

    mapping[timeIndex2] = np.array([np.arange(len(index2)), name2]).T

for timeIndex2, values in mapping.items():
    sub_dataframe = np.array(dataFrame.loc[dataFrame['time'] == timeIndex2, 'cell'])
    real_cell_name = mapping[timeIndex2][:, 1][sub_dataframe-1]
    dataFrame.loc[dataFrame['time'] == timeIndex2, 'cell'] = real_cell_name

dataFrame.to_csv('./Output_table_final.csv')


"""
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
    locations_cells[mask_cluster][::10,2],
    locations_cells[mask_cluster][::10,1],
    locations_cells[mask_cluster][::10,0],
    c=labelsFitted[mask_cluster][::10],
    s=3,cmap='tab20',alpha=0.15)
ax.scatter(centers[:,2],centers[:,1],centers[:,0],color='b',alpha=1)
ax.scatter(spots_locations[:,2],spots_locations[:,1],spots_locations[:,0],color='r',alpha=0.5,s=25)
ax.scatter((output['spot_z']-analyzis.imageData.sizes.z/2)*analyzis.imageData.slicePixelRatio,output['spot_y']-analyzis.imageData.sizes.y/2,output['spot_x']-analyzis.imageData.sizes.x/2,color='k',alpha=1,s=5)
ax.scatter((spot_table['spot_z']-analyzis.imageData.sizes.z/2)*analyzis.imageData.slicePixelRatio,spot_table['spot_y']-analyzis.imageData.sizes.y/2,spot_table['spot_x']-analyzis.imageData.sizes.x/2,color='purple',alpha=1,s=5)

plt.xlim(-256, 256)
plt.show()

ax = plt.figure().add_subplot(projection='3d')
ax.scatter(
    locations_cells[mask_cluster][::10,2],
    locations_cells[mask_cluster][::10,1],
    locations_cells[mask_cluster][::10,0],
    c=labelsFitted[mask_cluster][::10],
    s=3,cmap='tab20',alpha=0.15)
ax.scatter(centers[:,2],centers[:,1],centers[:,0],color='b',alpha=1)
ax.scatter(spots_locations[:,2],spots_locations[:,1],spots_locations[:,0],color='r',alpha=0.5,s=25)
ax.scatter((output['spot_z']-analyzis.imageData.sizes.z/2)*analyzis.imageData.slicePixelRatio,output['spot_y']-analyzis.imageData.sizes.y/2,output['spot_x']-analyzis.imageData.sizes.x/2,color='k',alpha=1,s=5)
ax.scatter((spot_table['spot_z']-analyzis.imageData.sizes.z/2)*analyzis.imageData.slicePixelRatio,spot_table['spot_y']-analyzis.imageData.sizes.y/2,spot_table['spot_x']-analyzis.imageData.sizes.x/2,color='purple',alpha=1,s=5)

plt.xlim(-256, 256)
plt.show()


newTime = time()
print("--- %s seconds ---" % (newTime - startTime))

for slice in range(70):
    plt.figure(figsize=(12,12))
    plt.imshow(voxel_map_bulle[slice],alpha=1,cmap='Greys_r')
    plt.imshow(voxelNormalized[slice],alpha=0.8,vmin=0,vmax=np.mean(all_tresh))
    plt.scatter(
        spot_table.loc[spot_table['spot_z']==slice,'spot_y'],
        spot_table.loc[spot_table['spot_z']==slice,'spot_x'],
        color='r',s=1,alpha=0.1)
    plt.scatter(
        output.loc[output['spot_z']==slice,'spot_y'],
        output.loc[output['spot_z']==slice,'spot_x'],
        color='b',s=1,alpha=1)

    # plt.scatter(spots[2][spots[0]==slice],spots[1][spots[0]==slice],color='r',s=1,alpha=0.1)
"""
