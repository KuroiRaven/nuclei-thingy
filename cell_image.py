import getopt
import glob as glob

import czifile
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from tqdm import tqdm

# =============================================================================
# FUNCTIONS
# =============================================================================

def myfmad(array,axis=0,sigma_conv=True):
    """"""
    if axis == 0:
        step = abs(array-np.nanmedian(array,axis=axis))
    else:
        step = abs(array-np.nanmedian(array,axis=axis)[:,np.newaxis])
    return np.nanmedian(step,axis=axis)*[1,1.48][int(sigma_conv)]


def rm_outliers(array, m=1.5, kind='sigma',axis=0, return_borders=False):
    if type(array)!=np.ndarray:
        array=np.array(array)
    
    if m!=0:
        array[array==np.inf] = np.nan
        #array[array!=array] = np.nan
        
        if kind == 'inter':
            interquartile = np.nanpercentile(array, 75, axis=axis) - np.nanpercentile(array, 25, axis=axis)
            inf = np.nanpercentile(array, 25, axis=axis)-m*interquartile
            sup = np.nanpercentile(array, 75, axis=axis)+m*interquartile            
            mask = (array >= inf)&(array <= sup)
        if kind == 'sigma':
            sup = np.nanmean(array, axis=axis) + m * np.nanstd(array, axis=axis)
            inf = np.nanmean(array, axis=axis) - m * np.nanstd(array, axis=axis)
            mask = abs(array-np.nanmean(array, axis=axis)) <= m * np.nanstd(array, axis=axis)
        if kind =='mad':
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

# =============================================================================
# TABLE
# =============================================================================

img = czifile.imread('/Users/cretignier/Downloads/Bursting trial_34.czi')

sizex = np.shape(img)[4]
sizey = np.shape(img)[5]
sizez = np.shape(img)[3]

time_index = 10
cutoff_sphere_z = 5 #fixed with the histogram distribution
cutoff_spot_z = 40 #fixed with the histogram distribution
slice_conversion = 2 #to be fixed
tresh_volume = 500
tresh_dist = np.sqrt(3)
k_cluster = 80 
radius_cell_pxl = 7.5
vicinity = 2

voxel = img[0,time_index,0,:,:,:,0]
voxel2 = []
for v in voxel:
    inter = v-np.nanmedian(v)
    inter/=myfmad(inter)
    voxel2.append(inter)
voxel2 = np.array(voxel2)

#x,y = np.histogram(np.ravel(voxel2[40]),bins=100)
#plt.plot(x,y)

tresh = voxel2>cutoff_sphere_z

locations = []
for j in range(len(voxel2)):
    x,y = np.where(tresh[j])
    tab = np.array([x,y,j*np.ones(len(x))]).T
    locations.append(tab)
locations = np.vstack(locations)

locations[:,0]-=sizex/2
locations[:,1]-=sizey/2
locations[:,2]-=sizez/2
locations[:,2]*=slice_conversion

borders = np.array_split(np.arange(len(locations)),5000)
all_min_dist = []
for index in tqdm(borders):
    sub_table = locations[index]
    dist = (sub_table[:,0]-locations[:,0][:,np.newaxis])**2+(sub_table[:,1]-locations[:,1][:,np.newaxis])**2+(sub_table[:,2]-locations[:,2][:,np.newaxis])**2
    dist = np.where(dist==0,np.nan,dist)
    dist = np.nanmin(dist,axis=0)
    all_min_dist.append(dist)
all_min_dist = np.array(all_min_dist)
all_min_dist = np.sqrt(np.hstack(all_min_dist))

locations_cells = locations[all_min_dist<=tresh_dist]

df = pd.DataFrame(locations_cells,columns=['x','y','z'])

kmeans = KMeans(n_clusters=k_cluster, init='k-means++', random_state=0).fit(df)
centers_fitted = kmeans.cluster_centers_
labels_fitted = kmeans.labels_

standard_cluster = []
radius_cluster = []
for j in np.unique(labels_fitted):
    pos_cluster = locations_cells[labels_fitted==j]
    cluster_center = kmeans.cluster_centers_[j]
    std_pts = np.sum((pos_cluster-cluster_center)**2,axis=1)
    radius_cluster.append(np.median(np.sqrt(std_pts)))
    standard_cluster.append(np.sqrt(np.sum(std_pts))/len(pos_cluster))
standard_cluster = np.array(standard_cluster)
radius_cluster = np.array(radius_cluster)

nb_points = np.array([np.sum(labels_fitted==i) for i in np.sort(np.unique(labels_fitted))])

mask1 = (nb_points>tresh_volume)
mask2 = rm_outliers(standard_cluster)[0]
mask = mask1&mask2
labels_kept = np.arange(len(nb_points))[mask]

mask_cluster = np.in1d(labels_fitted,labels_kept)
centers_fitted_kept = centers_fitted[mask]

dist_cluster = np.sqrt((centers_fitted_kept[:,0]-centers_fitted_kept[:,0][:,np.newaxis])**2+(centers_fitted_kept[:,1]-centers_fitted_kept[:,1][:,np.newaxis])**2+(centers_fitted_kept[:,2]-centers_fitted_kept[:,2][:,np.newaxis])**2)
tresh_dist_cluster = dist_cluster<(radius_cell_pxl*2)
rows,cols = np.where(tresh_dist_cluster)
cluster_twin = (rows>cols)
rows = rows[cluster_twin]
cols = cols[cluster_twin]

for r,c in zip(rows,cols):
    labels_fitted[labels_fitted==labels_kept[r]] = labels_kept[c]


final_labels = np.unique(labels_fitted[mask_cluster])
print('[INFO] Number of cells detected : %.0f'%(len(final_labels)))

centers = []
for j in final_labels:
    centers.append(np.mean(locations_cells[labels_fitted==j],axis=0))
centers = np.array(centers)

centers_bulle = centers.copy()

centers_bulle[:,2]/=slice_conversion
centers_bulle[:,0]+=sizex/2
centers_bulle[:,1]+=sizey/2
centers_bulle[:,2]+=sizez/2


x = np.arange(sizex)*np.ones(sizey)[:,np.newaxis]*np.ones(sizez)[:,np.newaxis][:,:,np.newaxis]
y = np.ones(sizex)*np.arange(sizey)[:,np.newaxis]*np.ones(sizez)[:,np.newaxis][:,:,np.newaxis]
z = np.ones(sizex)*np.ones(sizey)[:,np.newaxis]*np.arange(sizez)[:,np.newaxis][:,:,np.newaxis]


spot_table = []
all_tresh = []
voxel_map_bulle = np.zeros(np.shape(voxel2)).astype(bool)
for counter in np.arange(len(centers_bulle)):
    bulle = centers_bulle[counter]
    mask_bulle = np.sqrt((x-bulle[1])**2+(y-bulle[0])**2+((z-bulle[2])*slice_conversion)**2)<(1.5*radius_cell_pxl)
    voxel_map_bulle = voxel_map_bulle|mask_bulle
    val_sup = rm_outliers(voxel2[mask_bulle],kind='inter',return_borders=True)[2]
    all_tresh.append(val_sup)
    spotz,spotx,spoty = np.where((voxel2>=val_sup)&(mask_bulle))
    spoti = np.array([voxel2[zi,xi,yi] for xi,yi,zi in zip(spotx,spoty,spotz)])
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
    cube = voxel2[zi-vicinity:zi+vicinity+1,xi-vicinity:xi+vicinity+1,yi-vicinity:yi+vicinity+1]
    loc = np.where(cube==np.max(cube))
    crit = np.sum([abs(l-vicinity) for l in loc])
    loc_max.append(int(crit==0))
spot_table['loc_max_qc'] = np.array(loc_max)
output = spot_table.loc[spot_table['loc_max_qc']==1]

voxel_map_bulle = np.transpose(voxel_map_bulle,(0,2,1))

voxel_spots = (voxel2>cutoff_spot_z)&(voxel_map_bulle)
spots = np.where(voxel_spots)

spots_locations = (np.array([spots[1],spots[2],spots[0]]).T).astype('float')

spots_locations[:,0]-=sizex/2
spots_locations[:,1]-=sizey/2
spots_locations[:,2]-=sizez/2
spots_locations[:,2]*=slice_conversion

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(
    locations_cells[mask_cluster][::10,2],
    locations_cells[mask_cluster][::10,1],
    locations_cells[mask_cluster][::10,0],
    c=labels_fitted[mask_cluster][::10],
    s=3,cmap='tab20',alpha=0.15)
ax.scatter(centers_fitted_kept[:,2],centers_fitted_kept[:,1],centers_fitted_kept[:,0],color='g',alpha=1)
ax.scatter(centers[:,2],centers[:,1],centers[:,0],color='b',alpha=1)
ax.scatter(spots_locations[:,2],spots_locations[:,1],spots_locations[:,0],color='r',alpha=0.5,s=25)
ax.scatter((output['spot_z']-sizez/2)*slice_conversion,output['spot_y']-sizey/2,output['spot_x']-sizex/2,color='k',alpha=1,s=5)

plt.xlim(-256,256)

for slice in range(70):
    plt.figure()
    plt.imshow(voxel_map_bulle[slice],alpha=1,cmap='Greys_r')
    plt.imshow(voxel2[slice],alpha=0.8,vmin=0,vmax=np.mean(all_tresh))
    plt.scatter(
        spot_table.loc[spot_table['spot_z']==slice,'spot_y'],
        spot_table.loc[spot_table['spot_z']==slice,'spot_x'],
        color='r',s=1,alpha=0.1)
    plt.scatter(
        output.loc[output['spot_z']==slice,'spot_y'],
        output.loc[output['spot_z']==slice,'spot_x'],
        color='b',s=1,alpha=1)

    #plt.scatter(spots[2][spots[0]==slice],spots[1][spots[0]==slice],color='r',s=1,alpha=0.1)






