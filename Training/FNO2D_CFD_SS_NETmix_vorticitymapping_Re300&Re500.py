# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from basics import SpectralConv2d
import matplotlib.pyplot as plt
from losses import LpLoss
from functools import reduce
import operator

import time
from timeit import default_timer

class FNN2d(nn.Module):
    def __init__(self, modes1, modes2,
                 width=64, fc_dim=128,
                 layers=None,
                 in_dim=3, out_dim=1,
                 activation='tanh',
                 pad_x=0, pad_y=0):
        super(FNN2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.padding = (0, 0, 0, pad_y, 0, pad_x)
        # input channel is 3: (a(x, y), x, y)
        if layers is None:
            self.layers = [width] * 4
        else:
            self.layers = layers
        self.fc0 = nn.Linear(in_dim, layers[0])

        self.sp_convs = nn.ModuleList([SpectralConv2d(
            in_size, out_size, mode1_num, mode2_num)
            for in_size, out_size, mode1_num, mode2_num
            in zip(self.layers, self.layers[1:], self.modes1, self.modes2)])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(self.layers, self.layers[1:])])

        self.fc1 = nn.Linear(layers[-1], fc_dim)
        self.fc2 = nn.Linear(fc_dim, out_dim)
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
            - x : (batch size, x_grid, y_grid, 2)
        Returns:
            - x: (batch size, x_grid, y_grid, 1)
        '''
        length = len(self.ws)
        batchsize = x.shape[0]
        nx, ny = x.shape[1], x.shape[2] # original shape
        x = F.pad(x, self.padding, "constant", 0)
        size_x, size_y = x.shape[1], x.shape[2]

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x.view(batchsize, self.layers[i], -1)).view(batchsize, self.layers[i+1], size_x, size_y)
            x = x1 + x2
            if i != length - 1:
                x = self.activation(x)
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = x.reshape(batchsize, size_x, size_y, self.out_dim)
        x = x[..., :nx, :ny, :]
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



def recover_grid(n,res):
    def expand_array(data,dim):
        # zrs=np.zeros((dim,data.shape[0],data.shape[1]))
        # aux=np.expand_dims(data, axis=0)
        # zrs[:,:,:]=aux
        return np.repeat(data[np.newaxis,...],dim,axis=0)
    
    x=np.linspace(0,1,res)
    y=np.linspace(0,1,res)
    x_grid,y_grid=np.meshgrid(x,y)
    x_grid=expand_array(x_grid,n)
    y_grid=expand_array(y_grid,n)
    
    return x_grid, y_grid
    

def shift_data(data,t_sim,t_shift,dt):
    dt_num=int(dt*data.shape[0]/t_sim)
    x=np.zeros(((data.shape[0]-int(dt_num*t_shift)),data.shape[1],data.shape[2],t_shift))
    y=np.zeros(((data.shape[0]-int(dt_num*t_shift)),data.shape[1],data.shape[2]))
    for t in range(t_shift):
        x[:,:,:,t]=data[t*dt_num:(data.shape[0]-(t_shift-t)*dt_num),:,:]
    y=data[t_shift*dt_num:,:,:]
    return x,y

def predict(model,data_x):
    model.eval()
    x=data_x
    with torch.no_grad():
        x=x.to(device, dtype=torch.float32)
    return model(x)

def visualise_data(NN_pred,y_arr,tpred,tstep,variable):
    NN_pred=NN_pred.cpu().detach().numpy()
    y_arr=y_arr.cpu().detach().numpy()
    translate={"w":0      
        }
    plt.figure(num=1,figsize=(8, 8))
    plt.subplot(2,1,1)
    plt.contourf(NN_pred[tpred,:,:,translate[variable]], extent=[0, 1, 0, 1], origin='lower', cmap='coolwarm')
    plt.colorbar(label="predicted")
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.grid(True)
    plt.axis('equal')

    plt.subplot(2,1,2)
    plt.contourf(y_arr[tstep+tpred,:,:,translate[variable]], extent=[0, 1, 0, 1], origin='lower', cmap='coolwarm')
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
    
def load_checkpoint(model,optimizer,path,device):
    ckpt=torch.load(path,map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print("Model and optimizer sucessfully loaded!")

def save_testdata(train_avg,train_std,test_avg,test_std,file_name):
    to_save=np.stack((train_avg,train_std,test_avg,test_std),axis=-1)
    np.savetxt(file_name+".csv",to_save,delimiter=",")
    print("Test data was saved!")


def count_params(model):
    c=0
    for p in model.parameters():
        c+=reduce(operator.mul,list(p.size()))
    return c

def produce_input(data,grid_x,grid_y):
    aux=torch.zeros((1,data.shape[1],data.shape[2]))
    aux[:,:,:]=data[...,0]
    return torch.stack([aux,grid_x,grid_y],dim=-1)

with torch.no_grad():
    torch.cuda.empty_cache()



## Loading data
values_mask=np.load("D:/O4W/Simulations/PINO/FNO_NETmix/Re300/data_grid_mask.npy")
y_w_train=np.load("D:/O4W/Simulations/PINO/FNO_SS/CFD/data_hf_train_Re300&Re500.npy")
y_w_test=np.load("D:/O4W/Simulations/PINO/FNO_SS/CFD/data_hf_test_Re300&Re500.npy")
x_w_train=np.load("D:/O4W/Simulations/PINO/FNO_SS/CFD/data_lf_train_Re300&Re500.npy")
x_w_test=np.load("D:/O4W/Simulations/PINO/FNO_SS/CFD/data_lf_test_Re300&Re500.npy")

# ## Shift data
# t_sim=2
# t_shift=2
# dt=0.1
# x_w,y_w=shift_data(data_grid_w[:5000,...],t_sim,t_shift,dt)

## Recover grid
grid_x_train, grid_y_train=recover_grid(x_w_train.shape[0],x_w_train.shape[1])
grid_x_test, grid_y_test=recover_grid(x_w_test.shape[0],x_w_test.shape[1])


## Convert arrays into PyTorch tensor
x_w_train,y_w_train=torch.from_numpy(x_w_train),torch.from_numpy(y_w_train)
x_w_test,y_w_test=torch.from_numpy(x_w_test),torch.from_numpy(y_w_test)
grid_x_train,grid_y_train=torch.from_numpy(grid_x_train),torch.from_numpy(grid_y_train)
grid_x_test,grid_y_test=torch.from_numpy(grid_x_test),torch.from_numpy(grid_y_test)
values_mask=torch.from_numpy(values_mask)

## Create dataset

batch_size=5
# ntrain=int(0.9*x_w.shape[0])
data_x_train=torch.stack([x_w_train,grid_x_train,grid_y_train],dim=-1)
data_x_test=torch.stack([x_w_test,grid_x_test,grid_y_test],dim=-1)
data_y_train=torch.stack([y_w_train],dim=-1)
data_y_test=torch.stack([y_w_test],dim=-1)

## Create data loaders
train_loader=DataLoader(torch.utils.data.TensorDataset(data_x_train, data_y_train),batch_size=batch_size,shuffle=True)
test_loader=DataLoader(torch.utils.data.TensorDataset(data_x_test, data_y_test),batch_size=batch_size,shuffle=False)


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
model= FNN2d(modes1=[32,32,32,32],
              modes2=[32,32,32,32],
              fc_dim=128,
              layers=[64,64,64,64],
              in_dim=3,
              out_dim=1,
              activation="gelu"  
        ).to(device)

## Define optimizer
learning_rate=(2.5/2**0)*1e-6
# step_size=int(6*32/5)
milestones=[5,15,25,35,45]
gamma=0.5
optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

## Define loss function
loss_fn=LpLoss(size_average=True)

# #Load checkpoint
# load_checkpoint(model,optimizer,"FNO2D_CFD_SS_Re300_5000p_w_DR0.9_epoch70.pth", device)


## Training loop
epoch=70
all_avg_train_data_loss=[]
all_std_train_data_loss=[]
all_avg_test_data_loss=[]
all_std_test_data_loss=[]
start_time=time.time()
print("Starting the training loop.\n")
for t in range(epoch):
    print(f"Epoch {t+1}\n-------")
    ## Train
    train_data_loss=[]
    t1=default_timer()
    size=len(train_loader.dataset)
    current=0
    print(f"Size: {size}")
    for  batch, (X, y) in enumerate(train_loader):
            X=X.to(device, dtype=torch.float32) # Move input data to the device
            y=y.to(device,dtype=torch.float32)
            out=model(X)
            data_loss=compute_data_loss(out,y,mask=values_mask[X.shape[0],...],use_sum=True) # Compute the loss
            #optimizer.zero_grads() # Zero gradients
            data_loss.backward() # Backpropagate
            optimizer.step() # Update weights
            loss_value=data_loss.item()
            train_data_loss.append(loss_value)
            current+=X.shape[0]
            print(f"Loss: {loss_value:>7f} [{current:>5d}/{size:>5d}]")
    avg_train_data_loss=np.mean(train_data_loss)
    std_train_data_loss=np.std(train_data_loss)
    all_avg_train_data_loss.append(avg_train_data_loss.copy())
    all_std_train_data_loss.append(std_train_data_loss.copy())
          
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
            data_loss=compute_data_loss(out,y,mask=values_mask[X.shape[0],...],use_sum=True).item()
            test_data_loss.append(data_loss)
        avg_test_data_loss=np.mean(test_data_loss)
        std_test_data_loss=np.std(test_data_loss)
                
        all_avg_test_data_loss.append(avg_test_data_loss.copy())
        all_std_test_data_loss.append(std_test_data_loss.copy())              
        t2=default_timer()
             
        scheduler.step()
    print(f"Test average data loss: {avg_test_data_loss:>8f},\nTest std data loss: {std_test_data_loss:>8f},\nTrain average data loss: {avg_train_data_loss:>8f},\nTrain std data loss: {std_train_data_loss:>8f},\nTime (s): {t2-t1:.3f} \n")
print("Training complete!")
elapsed_time=time.time()-start_time
print(f"Training time (s): {elapsed_time:.3f}\n")
        
save_checkpoint("FNO2D_CFD_SS_Re300_5000p_w_DR0.9_Re300&Re500_epoch70", model, optimizer)
save_testdata(all_avg_train_data_loss,all_std_train_data_loss,all_avg_test_data_loss,all_std_test_data_loss,"FNO2D_CFD_SS_Re300_5000p_w_DR0.9_Re300&Re500_epoch70_testdata")

tstep=420
model_prediction=predict(model,data_x_test[tstep:(tstep+1),:,:,:])
visualise_data(model_prediction,data_x_test,0,tstep,"w")
visualise_test_error_plot(all_avg_test_data_loss,epoch)
