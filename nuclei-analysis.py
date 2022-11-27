import glob as glob
from time import time

import matplotlib
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from tqdm import tqdm

from models.analysis import Analysis
from models.returnValueThread import ReturnValueThread
from services.utils import match_nearest_ndim, rm_outliers

matplotlib.use('Qt5Agg')

pathFile = '../Bursting trial_34.czi'

startTime = time()

analyzis = Analysis(pathFile)

time_index = 10

cutoff_sphere_z = 5  # fixed with the histogram distribution // Intensité du noyau
cutoff_spot_z = 40  # fixed with the histogram distribution // Intensité des spots
tresh_volume = 500  # minimum element for a nucleus to be detected as a nucleus
tresh_dist = np.sqrt(3)  # minimum distance for the nearest neighbor to be valid
k_cluster = 80  # amount of nucleus we're looking for
radius_cell_pxl = 7.5  # radius of a nucleus in pxl
vicinity = 2 #proximity

time_indexes = [10,11,12]
for time_index in time_indexes:
    frame = analyzis.frames[time_index]
    voxel = frame.voxel
    voxelNormalized = frame.getNormalizedVoxel()

    #x,y = np.histogram(np.ravel(voxelNormalized[40]),bins=100)
    # plt.plot(x,y)

    tresh = voxelNormalized > cutoff_sphere_z


    def getLocations(slideId: int):
        x, y = np.where(tresh[slideId])  # all coordinates that are preserved after masking
        tab = np.array([x, y, slideId*np.ones(len(x))]).T
        return tab


    locations = np.array(list(map(getLocations, range(len(voxelNormalized)))))
    locations = np.vstack(locations)

    # Recenter the axes + Z axes converted to pixel
    locations[:, 0] -= analyzis.imageData.sizes.x/2
    locations[:, 1] -= analyzis.imageData.sizes.y/2
    locations[:, 2] -= analyzis.imageData.sizes.z/2
    locations[:, 2] *= analyzis.imageData.slicePixelRatio

    # Removes all noises
    borders = np.array_split(np.arange(len(locations)), 5000)


    def getDists(val):
        sub_table = locations[val]
        dist = (sub_table[:, 0]-locations[:, 0][:, np.newaxis])**2+(sub_table[:, 1] -
                                                                    locations[:, 1][:, np.newaxis])**2+(sub_table[:, 2]-locations[:, 2][:, np.newaxis])**2
        dist = np.where(dist == 0, np.nan, dist)
        dist = np.nanmin(dist, axis=0)
        return dist


    threadedDists = [ReturnValueThread(target=getDists, args=[val]) for val in borders]
    threadedDistsStarted = [thread.start() for thread in threadedDists]
    allMinDistThreaded = [thread.join() for thread in threadedDists]

    allMinDist = np.array(allMinDistThreaded)
    allMinDist = np.sqrt(np.hstack(allMinDist))
    # actual noise removal
    locations_cells = locations[allMinDist <= tresh_dist]

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

    spot_table = []
    all_tresh = []
    voxel_map_bulle = np.zeros(np.shape(voxelNormalized)).astype(bool)
    for counter in np.arange(len(centers_bulle)):
        bulle = centers_bulle[counter]
        mask_bulle = np.sqrt((x-bulle[1])**2+(y-bulle[0])**2+((z-bulle[2])*analyzis.imageData.slicePixelRatio)**2)<(1.5*radius_cell_pxl)
        voxel_map_bulle = voxel_map_bulle|mask_bulle
        val_sup = rm_outliers(voxelNormalized[mask_bulle],kind='inter',return_borders=True)[2]
        all_tresh.append(val_sup)
        spotz,spotx,spoty = np.where((voxelNormalized>=val_sup)&(mask_bulle))
        spoti = np.array([voxelNormalized[zi,xi,yi] for xi,yi,zi in zip(spotx,spoty,spotz)])
        print('[INFO] Treshold set to I_lim = %.2f for spot in nucleus %.0f. Spots nb = %.0f'%(val_sup,counter,len(spotx)))
        spotb = np.ones(len(spotz))*(counter+1)
        spott = np.ones(len(spotz))*time_index
        spotcx = np.ones(len(spotz))*bulle[0]
        spotcy = np.ones(len(spotz))*bulle[1]
        spotcz = np.ones(len(spotz))*bulle[2]
        spot_table.append(np.array([spoti,spotx,spoty,spotz,spotb,spotcx,spotcy,spotcz,spott]).T)
    spot_table = np.vstack(spot_table).astype('int')
    spot_table = pd.DataFrame(spot_table,columns=['spot_i','spot_x','spot_y','spot_z','cell','cell_x','cell_y','cell_z','time'])

    loc_max=[]
    for xi,yi,zi in zip(spot_table['spot_x'],spot_table['spot_y'],spot_table['spot_z']):
        cube = voxelNormalized[zi-vicinity:zi+vicinity+1,xi-vicinity:xi+vicinity+1,yi-vicinity:yi+vicinity+1]
        loc = np.where(cube==np.max(cube))
        crit = np.sum([abs(l-vicinity) for l in loc])
        loc_max.append(int(crit==0))
    spot_table['loc_max_qc'] = np.array(loc_max)
    output = spot_table.loc[spot_table['loc_max_qc']==1]

    spot_table.to_csv('./Output_table_time_index_%s.csv'%(str(time_index).zfill(3)))

all_files = np.sort(glob.glob('./Output_table_time_index*csv'))

for i,f in enumerate(all_files):
    f_temp = pd.read_csv(f,index_col=0)
    if not i:
        dataframe = f_temp.copy()
    else:
        dataframe = pd.concat([dataframe,f_temp])

mapping = {}
c=-1
for timeIndex1,timeIndex2 in zip(time_indexes[0:-1],time_indexes[1:]):
    c+=1
    uniqueVals1 = dataframe.loc[dataframe['time']==timeIndex1,['cell_x','cell_y','cell_z','cell']]
    uniqueVals2 = dataframe.loc[dataframe['time']==timeIndex2,['cell_x','cell_y','cell_z','cell']]
    uniqueCellsByTimeFrame1 = np.array(uniqueVals1.loc[~uniqueVals1.duplicated(subset=['cell']),['cell_x','cell_y','cell_z']])
    uniqueCellsByTimeFrame2 = np.array(uniqueVals2.loc[~uniqueVals2.duplicated(subset=['cell']),['cell_x','cell_y','cell_z']])

    dist_x = uniqueCellsByTimeFrame1[:,0]-uniqueCellsByTimeFrame2[:,0][:,np.newaxis]
    dist_y = uniqueCellsByTimeFrame1[:,1]-uniqueCellsByTimeFrame2[:,1][:,np.newaxis]
    dist_z = uniqueCellsByTimeFrame1[:,2]-uniqueCellsByTimeFrame2[:,2][:,np.newaxis]

    dist = np.sqrt((dist_x)**2+(dist_y)**2+(dist_z)**2)

    index1 = np.argmin(dist,axis=0)
    index2 = np.argmin(dist,axis=1)

    match2_to_1 = index1[index2] - np.arange(len(index2))
    match1_to_2 = index2[index1] - np.arange(len(index1))

    if not c:
        name1 = np.arange(len(index1))
        mapping[timeIndex1] = np.array([np.arange(len(index1)),name1]).T
    
    name2 = np.zeros(len(index2)).astype('int')
    name2[match2_to_1==0] = mapping[timeIndex1][:,1][index2[match2_to_1==0]]
    name2[match2_to_1!=0] = np.max(name2)+np.arange(sum(match2_to_1!=0))
        
    mapping[timeIndex2] = np.array([np.arange(len(index2)),name2]).T

for timeIndex,values in mapping.items():
    sub_dataframe = np.array(dataframe.loc[dataframe['time']==timeIndex,'cell'])
    real_cell_name = mapping[timeIndex][:,1][sub_dataframe-1]
    dataframe.loc[dataframe['time']==timeIndex,'cell'] = real_cell_name

dataframe.to_csv('./Output_table_final.csv')


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
    c=labels_fitted[mask_cluster][::10],
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

    #plt.scatter(spots[2][spots[0]==slice],spots[1][spots[0]==slice],color='r',s=1,alpha=0.1)
"""
