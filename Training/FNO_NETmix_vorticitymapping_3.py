# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from basics import SpectralConv3d
import matplotlib.pyplot as plt
from losses import LpLoss
from functools import reduce
import operator

import time
from timeit import default_timer

class FNN3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width=16, fc_dim=128, layers=None, in_dim=4, out_dim=1, activation='tanh', pad_x=0, pad_y=0, pad_z=0):
        '''
        Args:
            modes1: list of int, first dimension maximal modes for each layer
            modes2: list of int, second dimension maximal modes for each layer
            modes3: list of int, third dimension maximal modes for each layer
            layers: list of int, channels for each layer
            in_dim: int, input dimension
            out_dim: int, output dimension
        '''
        super(FNN3d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.padding = (0, 0, 0, pad_z, 0, pad_y, 0, pad_x)

        if layers is None:
            self.layers = [width] * 4
        else:
            self.layers = layers
        self.fc0 = nn.Linear(self.in_dim, layers[0])

        self.sp_convs = nn.ModuleList([SpectralConv3d(
            in_size, out_size, mode1_num, mode2_num, mode3_num)
            for in_size, out_size, mode1_num, mode2_num, mode3_num
            in zip(self.layers, self.layers[1:], self.modes1, self.modes2, self.modes3)])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(self.layers, self.layers[1:])])

        self.fc1 = nn.Linear(layers[-1], fc_dim)
        self.fc2 = nn.Linear(fc_dim, self.out_dim)
        
        if activation =='tanh':
            self.activation = F.tanh
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation == F.relu
        else:
            raise ValueError(f'{activation} is not supported')

    def forward(self, x):
        '''
        Args:
            x: (batchsize, x_grid, y_grid, t_grid, 3)
        Returns:
            u: (batchsize, x_grid, y_grid, t_grid, 1)
        '''
        length = len(self.ws)
        batchsize = x.shape[0]
        nx, ny, nz = x.shape[1], x.shape[2], x.shape[3]
        x = F.pad(x, self.padding, "constant", 0)
        size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]

        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x.view(batchsize, self.layers[i], -1)).view(batchsize, self.layers[i+1], size_x, size_y, size_z)
            x = x1 + x2
            if i != length - 1:
                # x = torch.tanh(x)
                x = self.activation(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        # x = torch.tanh(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = x.reshape(batchsize, size_x, size_y, size_z, self.out_dim) # make sure dimensions are what we expect before getting rid of padding
        x = x[..., :nx, :ny, :nz, :]
        return x

def compute_data_loss(s_pred, s_true,mask,use_sum=True):
    mask=mask.to(device, dtype=torch.float32)
    #print(s_pred[...,0].shape)
    p_pred = torch.mul(s_pred[..., 0],mask)
    #print(p_pred.shape)
    #u_pred =torch.mul(s_pred[..., 1],mask)
    #v_pred = torch.mul(s_pred[..., 2],mask)
    p_true = torch.mul(s_true[..., 0],mask)
    #print(p_true.shape)
    #u_true = s_true[..., 1]
    #v_true = s_true[..., 2]
        
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



def recover_grid(n_space,res_space,res_time,group,dist,t_sim):
    def expand_space(data,n_space,group):
        # data=np.expand_dims(data,axis=-1)
        # zrs_1=np.zeros((data.shape[0],data.shape[1],group))
        # zrs_1[...,:]=data
        # zrs_1=np.expand_dims(zrs_1,axis=0)
        # zrs_2=np.zeros((dim,data.shape[0],data.shape[1],group))
        # zrs_2[:,...]=zrs_1
        data=np.transpose(data,(2,0,1))
        data=np.repeat(data[...,np.newaxis],group,axis=-1)
        return data[:n_space,...]
    def expand_time(data,group,dist):
        data=np.transpose(data,(2,0,1))
        aux=np.zeros(((data.shape[0]-dist*(group-1)),data.shape[1],data.shape[2],group))
        for n in range(aux.shape[0]):
            aux[n,:,:,:]=np.transpose(data[n:(n+dist*(group-1)+1):dist,...],(1,2,0))
        return aux
        
        
    
    x=np.linspace(0,1,res_space)
    y=np.linspace(0,1,res_space)
    t=np.linspace(0,t_sim,res_time)
    
    x_grid,y_grid,t_grid=np.meshgrid(x,y,t)
    
    x_grid=expand_space(x_grid.copy(),n_space,group)
    y_grid=expand_space(y_grid.copy(),n_space,group)
    t_grid=expand_time(t_grid.copy(),group,dist)
    
    return x_grid, y_grid, t_grid
    

def reshape_arrays(w,m):
    lst=[]
    for i in [w,m]:
        aux=np.zeros((1,i.shape[0],i.shape[1],i.shape[2]))
        aux[0,:,:,:]=i
        aux=np.transpose(aux,(0,2,3,1))
        lst.append(aux)
    return lst[0], lst[1]



def group_data(data,group,dist):
    
    def produce_x(aux,data,group):
        for n in range(aux.shape[0]):
            aux[n,:,:,:]=np.expand_dims(data[n,...],axis=-1)
        return aux
    
    def produce_y(aux,data,group,dist):
        for n in range(aux.shape[0]):
            aux[n,:,:,:]=np.transpose(data[n:(n+dist*(group-1)+1):dist,...],(1,2,0))
        return aux
    
    aux=np.zeros(((data.shape[0]-dist*(group-1)),data.shape[1],data.shape[2],group))
    x=produce_x(aux.copy(),data,group)
    y=produce_y(aux.copy(),data,group,dist)
    
    return x,y
  

def predict(model,data_x):
    model.eval()
    x=data_x
    #print(x.shape)
    with torch.no_grad():
        x=x.to(device, dtype=torch.float32)
        #batch_size, S, S, T, nfields = x.shape
        #x_in=F.pad(x, (0, 0, 0, padding), "constant", 0)
        #out=model(x_in).reshape(batch_size, S, S, T + padding, nfields)
         #print(pred)
    return model(x)

def visualise_data(NN_pred,y_arr,tpred,group,tstep,variable):
    NN_pred=NN_pred.cpu().detach().numpy()
    y_arr=y_arr.cpu().detach().numpy()
    translate={"w":0      
        }
    plt.figure(num=1,figsize=(8, 8))
    plt.subplot(2,1,1)
    plt.contourf(NN_pred[tpred,:,:,group-1,translate[variable]], extent=[0, 1, 0, 1], origin='lower', cmap='coolwarm')
    plt.colorbar(label="predicted")
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.grid(True)
    plt.axis('equal')

    plt.subplot(2,1,2)
    plt.contourf(y_arr[tstep+tpred,:,:,group-1,translate[variable]], extent=[0, 1, 0, 1], origin='lower', cmap='coolwarm')
    plt.colorbar(label="reference")
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.grid(True)
    plt.axis('equal')

    plt.show()             

def visualise_test_error_plot(error,epoch):
    epoch_lst=range(1,epoch+1)
    # fig=plt.figure(num=2)
    # ax1=fig.add_subplot(111)
    plt.figure(num=2)
    plt.scatter(epoch_lst,error,c="k",marker="o",label="Data loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Average relative test L2 norm")
    plt.show()

def save_checkpoint(name, model, optimizer):
    checkpoint={
        "model":model.state_dict(),
        "optimizer":optimizer.state_dict(),
        }
    torch.save(checkpoint,name+".pth")
    print("Model and optimizer were saved!")

# def closure(out,y,mask_values):
#     optimizer.zero_grad()
#     loss=compute_loss(out,y,mask=values_mask[...,:X.shape[3]],use_sum=True)
#     loss.backward(retain_graph=True)
#     return loss
    
def load_checkpoint(model,optimizer,path,device):
    ckpt=torch.load(path,map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print("Model and optimizer sucessfully loaded!")

def count_params(model):
    c=0
    for p in model.parameters():
        c+=reduce(operator.mul,list(p.size()))
    return c


with torch.no_grad():
    torch.cuda.empty_cache()



## Loading data
values_mask=np.load("data_grid_mask.npy")
data_grid_w=np.load("data_grid_w_ext.npy")

## Group data
group=41
dist=25
x_w,y_w=group_data(data_grid_w[:5000,...],group,dist)

## Create spatio-temporal grid
t_sim=2
grid_x, grid_y, grid_t=recover_grid(x_w.shape[0],data_grid_w.shape[1],data_grid_w.shape[0]-1,group,dist,t_sim)


## Adjust dimensions
values_mask=np.expand_dims(values_mask[:group,...],axis=-1)
values_mask=np.transpose(values_mask,(3,1,2,0))


## Convert arrays into PyTorch tensor
x_w,y_w=torch.from_numpy(x_w),torch.from_numpy(y_w)
grid_x,grid_y,grid_t=torch.from_numpy(grid_x),torch.from_numpy(grid_y),torch.from_numpy(grid_t)
values_mask=torch.from_numpy(values_mask)

## Create dataset

batch_size=10
ntrain=int(0.9*x_w.shape[0])
data_x=torch.stack([x_w,grid_x,grid_y,grid_t],dim=-1)
data_y=torch.stack([y_w],dim=-1)

## Create data loaders
train_loader=DataLoader(torch.utils.data.TensorDataset(data_x[:ntrain,:,:,:,:], data_y[:ntrain,:,:,:]),batch_size=batch_size,shuffle=True)
test_loader=DataLoader(torch.utils.data.TensorDataset(data_x[ntrain:,:,:,:,:], data_y[ntrain:,:,:,:]),batch_size=batch_size,shuffle=False)


## Define device
device= torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"    
    )

# device=torch.device("cpu")()

## Define model
model= FNN3d(modes1=[12,12,12,12],
              modes2=[12,12,12,12],
              modes3=[12,12,12,12],
              fc_dim=128,
              layers=[32,32,32,32],
              in_dim=4,
              out_dim=1,
              activation="gelu"  
        ).to(device)

## Define optimizer
learning_rate=(5/2**0)*1e-5
# step_size=int(6*32/5)
milestones=[5]
gamma=0.5
optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=2.5e-5,max_lr=5e-4, step_size_up=step_size)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
# optimizer=torch.optim.LBFGS(model.parameters())

## Define loss function
loss_fn=LpLoss(size_average=True)

## Load checkpoint
# load_checkpoint(model,optimizer,"FNO3D_Re300_5000p_w_4rdmethod_2_epoch60.pth", device)


## Training loop
epoch=10
all_avg_test_data_loss=[]
start_time=time.time()
#padding=1
print("Starting the training loop.\n")
for t in range(epoch):
    print(f"Epoch {t+1}\n-------")
    ## Train
    t1=default_timer()
    size=len(train_loader.dataset)
    current=0
    print(f"Size: {size}")
    for  batch, (X, y) in enumerate(train_loader):
        X=X.to(device, dtype=torch.float32) # Move input data to the device
        y=y.to(device,dtype=torch.float32)
        out=model(X)
        data_loss=compute_data_loss(out,y,mask=values_mask[:X.shape[0],...],use_sum=True) # Compute the loss
        #optimizer.zero_grads() # Zero gradients
        data_loss.backward() # Backpropagate
        optimizer.step() # Update weights
        #scheduler.step()
        loss_value=data_loss.item()
        current+=X.shape[0]
        print(f"Loss: {loss_value:>7f} [{current:>5d}/{size:>5d}]")
    ## Test
    print("Testing...")
    size=len(test_loader.dataset)
    num_batches=len(test_loader)
    print(f"Number of test examples: {num_batches}")
    model.eval()
    test_data_loss=[]
    with torch.no_grad():
        for X,y in test_loader:
            X=X.to(device, dtype=torch.float32) # Move input data to the device
            y=y.to(device,dtype=torch.float32) # Move target labels to the device
            out=model(X)
            data_loss=compute_data_loss(out,y,mask=values_mask[:X.shape[0],...],use_sum=True).item()
            test_data_loss.append(data_loss)
        avg_test_data_loss=np.mean(test_data_loss)
                
        all_avg_test_data_loss.append(avg_test_data_loss.copy())
        t2=default_timer()
             
        scheduler.step()
    print(f"Test average data loss: {avg_test_data_loss:>8f}, time (s): {t2-t1:.3f} \n")
print("Training complete!")
elapsed_time=time.time()-start_time
print(f"Training time (s): {elapsed_time:.3f}\n")
        
save_checkpoint("FNO3D_Re300_5000p_w_PH4e-1_epoch10", model, optimizer)

tstep=250
model_prediction=predict(model,data_x[tstep:(tstep+2),:,:,:,:])
visualise_data(model_prediction,data_y,0,0,tstep,"w")
visualise_test_error_plot(all_avg_test_data_loss,epoch)
