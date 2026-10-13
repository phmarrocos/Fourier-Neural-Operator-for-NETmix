# -*- coding: utf-8 -*-
import numpy as np
import codecs
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from losses import LpLoss
import torch


## Unit cell properties ##
D=6.75e-3 #Chamber diameter, in m
L=2e-3 #Channel length, in m

def insert_points_outside_shape(shape_wall,all_points, num_rotations):
    def mod_vector(vec):
        return np.sqrt(vec.dot(vec))
    def calculate_cosv(x_axis,direction):
        return np.dot(x_axis,direction)/(mod_vector(x_axis)*mod_vector(direction))
    def calculate_angle(cosv,direction):
        if direction[1] >= 0:
            return np.arccos(cosv)
        else:
            return (2*np.pi)-np.arccos(cosv)        
    def calculate_sinv(cosv,direction):
        if direction[1] >= 0:
            return np.sqrt(1-cosv**2)
        else:
            return -np.sqrt(1-cosv**2)
    def find_regions(x_axis,wall,centroid):
        direction_NE=np.max(wall,axis=0)-centroid
        direction_NW=[np.min(wall,axis=0)[0],np.max(wall,axis=0)[1]]-centroid
        direction_SW=np.min(wall,axis=0)-centroid
        direction_SE=[np.max(wall,axis=0)[0],np.min(wall,axis=0)[1]]-centroid
        cosv_NE=calculate_cosv(x_axis, direction_NE)
        cosv_NW=calculate_cosv(x_axis, direction_NW)
        cosv_SW=calculate_cosv(x_axis, direction_SW)
        cosv_SE=calculate_cosv(x_axis, direction_SE)
        return calculate_angle(cosv_NE,direction_NE),calculate_angle(cosv_NW,direction_NW),calculate_angle(cosv_SW,direction_SW),calculate_angle(cosv_SE,direction_SE)
    
    def find_maximum_range(direction,cosv,NE,NW,SW,SE):
        angle=calculate_angle(cosv, direction)
        if (angle<NW and angle>NE) or (angle<SE and angle>SW):
            return (0.5/np.abs(calculate_sinv(cosv,direction)))-(mod_vector(direction))
        else:
            return (0.5/np.abs(cosv))-(mod_vector(direction))
    
    def regulate_boundaries(data):
        tol=0.02
        def verify_tolerance(x,tol,num):
            if (x<=num+tol) and (x>=num-tol):
                return True
            else:
                return False
            
        for idx, x in np.ndenumerate(data):
            if verify_tolerance(x,tol,0):
                data[idx]=0
            if verify_tolerance(x,tol,1):
                data[idx]=1
        return data        
        
    wall = shape_wall.copy() # To keep original points unchanged
    new_points_list = []

    # Calculate centroid of the original shape points (optional, but can be helpful for expansion direction)
    centroid = np.mean(all_points, axis=0)
    # Find x axis direction, centered on centroid
    x_axis=np.array([1,0.5])-centroid
    NE,NW,SW,SE=find_regions(x_axis, wall, centroid)
    
    
    for r in range(num_rotations):    
        for idx in range(wall.shape[0]):
            direction=wall[idx]-centroid
            cosv=calculate_cosv(x_axis,direction)
            sinv=calculate_sinv(cosv,direction)
            s=find_maximum_range(direction, cosv, NE, NW, SW, SE)
            dl=np.array([((num_rotations-r)/num_rotations)*s*cosv,(((num_rotations-r)/num_rotations))*s*sinv])
            new_points_list.append(wall[idx]+dl)
            
    return regulate_boundaries(np.array(new_points_list))
    

def transform_data(data,D,L):
    data_xy_trans=np.copy(data)
    data_xy_trans=data_xy_trans+(D+np.sqrt(2)*L/2)/2
    data_xy_trans=data_xy_trans/(D+np.sqrt(2)*L/2)
    return data_xy_trans        

def include_inlet_points(data):
    ind=np.array([[0,0],[0,1],
                 [1,0],[1,1]])
    ind=np.where(ind==[0,0],np.min(data,axis=0),ind)
    ind=np.where(ind==[1,1],np.max(data,axis=0),ind)
    ind=np.where(ind==[0,1],[np.min(data,axis=0)[0],np.max(data,axis=0)[1]],ind)
    ind=np.where(ind==[1,0],[np.max(data,axis=0)[0],np.min(data,axis=0)[1]],ind)
    return np.vstack((data,ind))

def insert_zeros_outside(data,dim):
    zrs=np.zeros([data.shape[0],dim])
    return np.hstack((data,zrs))

def expand_array(data,dim):
    zrs=np.zeros((dim,data.shape[0],data.shape[1]))
    for i in range(dim):
        zrs[i,:,:]=data
    return zrs

def compute_distances(domain,wall):
    def mod_vector(vec):
        return np.sqrt(vec.dot(vec))
    dists=[]
    xy_dists=[]
    for i in domain:
        for j in wall:
            dists.append(mod_vector(i-j))
        min_dists=np.array(min(dists))
        xy_dists.append(np.hstack((i,min_dists)))
        dists.clear()
        
    return np.array(xy_dists)

# def produce_mask(domain, outside):
#     values=np.isin(domain,outside)
#     return np.hstack((domain,~values))[:,0:3]

def produce_mask(data):
    output=(data!=0)
    return output

def produce_2D_map(data,mtd):
    assert data.shape[1] == 3
    x_coords=data[:,0]
    y_coords=data[:,1]
    func_val=data[:,2]

    grid_resolution = 100 # Number of points in each direction for the regular grid
    x_regular_grid = np.linspace(0, 1, grid_resolution)
    y_regular_grid = np.linspace(0, 1, grid_resolution)
    grid_x_regular, grid_y_regular = np.meshgrid(x_regular_grid, y_regular_grid)
    
    # Use griddata to interpolate the irregular data onto the regular grid
    # 'linear' interpolation is used here, other options are 'nearest', 'cubic'
    func_interpolated = griddata((x_coords, y_coords), func_val,
                                      (grid_x_regular, grid_y_regular), method=mtd)
    return func_interpolated

def produce_3D_map(data,mtd,s):
    translate={
        "w":2
        }
    assert s in translate
    maps=[]
    for i in range(data.shape[0]):
        maps.append(produce_2D_map(np.transpose(data[i,:,(0,1,translate[s])]), mtd))
        print(f"Progress on mapping {s}: {i/data.shape[0]:.2f}\n")
    return np.array(maps)
               
def standardize(data,mask):
    mean=np.mean(data,where=(mask==1))
    std=np.std(data,where=(mask==1))
    output=(data-mean)/std
    return output*mask, mean, std

def fill_outside_values(grid):
    
    def fill_outside(grid,rows,cols):
        copy_grid=grid.copy()
        for r in range(rows):
            for c in range(cols):
                if grid[r,c]==0:
                    neighbors = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
                    values=0
                    n_values=0
                    for nr, nc in neighbors:
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != 0:
                            n_values+=1
                            values+=grid[nr, nc]
                    if n_values!=0:
                        copy_grid[r, c]=values/n_values
        if not np.all(copy_grid):
            return fill_outside(copy_grid,rows,cols) 
        else:
            return copy_grid
    
    rows,cols=grid.shape[1],grid.shape[2]
    grids=[]
    for i in range(grid.shape[0]):
        grids.append(fill_outside(grid[i,:,:],rows,cols))
        print(f"Progress on extending grid: {i/grid.shape[0]:.2f}\n")
    return np.array(grids)
    
def compute_data_loss(s_pred, s_true,mask,use_sum=True):
    #print(s_pred[...,0].shape)
    p_pred = torch.mul(s_pred,mask)
    #print(p_pred.shape)
    p_true = torch.mul(s_true,mask)
    #print(p_true.shape)
        
    lploss = LpLoss(size_average=True)
    loss_p = lploss(p_pred, p_true)
    #loss_u = lploss(u_pred, u_true)
    #loss_v = lploss(v_pred, v_true)
    #loss_s = torch.stack([loss_p, loss_u, loss_v], dim=-1)
    loss_s=loss_p
    if use_sum:
        data_loss = torch.sum(loss_s)
    else:
        data_loss = torch.mean(loss_s)
    return data_loss


# with codecs.open("NETmixUnitCell_ES1e-4_everything_2", encoding="utf-8-sig") as f:
#     data_all_points=np.loadtxt(f, delimiter=",", skiprows=1)

data_all_points=np.load("D:/O4W/Simulations/PINO/NETmix_Data/Re500/Co0p1/PINO_Raw_Data_Re500_vorticity_5000p.npy")
with codecs.open("D:/O4W/Simulations/PINO/FNO_NETmix/Re300/NETmixUnitCell_ES1e-4_wall_3", encoding="utf-8-sig") as f:
    data_all_wall=np.loadtxt(f, delimiter=",", skiprows=1)
        

    
data_xy=data_all_points[:,:,1:3]
data_xy_wall=data_all_wall[:,1:3]
data_xy_trans=transform_data(data_xy, D, L)
data_all_trans=np.concatenate((data_xy_trans,data_all_points[:,:,3:]),axis=2)

## Reduce discretization
r_data=0.1
idx=np.random.choice(range(data_all_trans.shape[1]),size=int(data_all_trans.shape[1]*r_data),replace=False)
data_r_trans=data_all_trans[:,idx,:]


data_xy_wall_trans=transform_data(data_xy_wall, D, L)
#data_xy_wall_trans=include_inlet_points(data_xy_wall_trans)
data_xy_outside=insert_points_outside_shape(data_xy_wall_trans, data_xy_trans[0,:,:], 4)
data_all_outside=insert_zeros_outside(data_xy_outside,(data_all_trans.shape[2]-2))
data_all_outside=expand_array(data_all_outside, data_all_trans.shape[0])
data_xy_domain=np.concatenate((data_all_trans,data_all_outside),axis=1)
data_r_domain=np.concatenate((data_r_trans,data_all_outside),axis=1)


data_grid_w=produce_3D_map(data_r_domain,"linear","w")
data_grid_mask=produce_mask(data_grid_w)
# data_grid_mask=expand_array(data_grid_mask, data_all_trans.shape[0])

data_grid_w_ext,mean_w,std_w=standardize(data_grid_w,data_grid_mask)
standardized_w_data=np.array([[mean_w],
                   [std_w]
                   ])

data_grid_w_ext=fill_outside_values(data_grid_w_ext)

## Compute loss
y=np.load("D:/O4W/Simulations/PINO/FNO_NETmix/Re300/data_grid_w_ext.npy")
y,x,m=torch.from_numpy(y),torch.from_numpy(data_grid_w_ext),torch.from_numpy(data_grid_mask)
loss=compute_data_loss(x,y,mask=m[0,...],use_sum=True).item()

# np.save("data_grid_w",data_grid_w)
np.save("data_grid_w_ext_DiscretizationReduction0.9_Re500",data_grid_w_ext)
np.save("standardized_w_data_DiscretizationReduction0.9_Re500",standardized_w_data)
np.save("data_grid_mask_DiscretizationReduction0.9_Re500",data_grid_mask)
# np.save("data_grid_dist",data_grid_dist)



# ## Plotting to visualize

plt.figure(num=1,figsize=(8, 8))
plt.scatter(data_r_trans[0,:,0], data_r_trans[0,:,1], color='blue', label='All points')
plt.scatter(data_xy_wall_trans[:,0], data_xy_wall_trans[:,1], color='red', label='Wall points')
plt.scatter(data_xy_outside[:,0], data_xy_outside[:,1], color='black', label='Outside points')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.xlim([0,1])
plt.ylim([0,1])
plt.legend()
plt.grid(True)
plt.axis('equal') # Equal aspect ratio for x and y axes

plt.figure(num=2,figsize=(8, 8))
plt.contourf(data_grid_w_ext[3500,...], extent=[0, 1, 0, 1], origin='lower', cmap='coolwarm')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.xlim([0,1])
plt.ylim([0,1])
plt.grid(True)
plt.axis('equal')

# # plt.figure(num=2,figsize=(8, 8))
# # sct=plt.scatter(data_xy_dist[:,0], data_xy_dist[:,1],c=data_xy_dist[:,2], cmap="coolwarm")
# # plt.colorbar(sct,label="Distance")
# # plt.xlabel('X-coordinate')
# # plt.ylabel('Y-coordinate')
# # plt.xlim([0,1])
# # plt.ylim([0,1])
# # plt.grid(True)
# # plt.axis('equal')

# plt.figure(num=3,figsize=(8, 8))
# plt.subplot(1,2,1)
# plt.imshow(data_grid_w[0,:,:], extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
# plt.colorbar(label="Mask")
# plt.xlabel('X-coordinate')
# plt.ylabel('Y-coordinate')
# plt.xlim([0,1])
# plt.ylim([0,1])
# plt.grid(True)
# plt.axis('equal')

# # plt.subplot(1,2,2)
# # plt.imshow(data_grid_dist[0,:,:], extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
# # plt.colorbar(label="Distances")
# # plt.xlabel('X-coordinate')
# # plt.ylabel('Y-coordinate')
# # plt.xlim([0,1])
# # plt.ylim([0,1])
# # plt.grid(True)
# # plt.axis('equal')


# # plt.show()
