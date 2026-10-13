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


class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        
        ## Define convolutions which maintain the x and y dimensions
        self.conv_op=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
            nn.GELU()
            )
    def forward(self,x):
        return self.conv_op(x)
    
class DownSample_2D(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv=DoubleConv(in_channels,out_channels)
        
        ## Define max pooling which reduces x and y dimensions by 2
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        
    def forward(self,x):
        down=self.conv(x)
        p=self.pool(down)
        
        return down,p

class UpSample_2D(nn.Module):
    def __init__(self,in_channels,out_channels,upscale_dim=True):
        super().__init__()
        
        if upscale_dim:
            ## Define transpose convolution which upscales x and y dimensions by 2
            self.up=nn.ConvTranspose2d(in_channels,in_channels//2,kernel_size=2,stride=2)
        else:
            ## Define convolution which maintains x and y dimensions, though reduces channels by 2
            self.up=nn.Conv2d(in_channels,in_channels//2,kernel_size=3,padding=1)
        self.conv=DoubleConv(in_channels,out_channels)
        
    def forward(self,x1,x2):
        x1=self.up(x1)
        
        ## Define concatanation to enable skip connections
        x=torch.cat([x1,x2],1)
        
        return self.conv(x)

class UNet_2D(nn.Module):
    def __init__(self,in_channels,num_classes):
        super(UNet_2D,self).__init__()
        
        ## Define coder convolution blocks
        self.down_conv_1=DownSample_2D(in_channels,32)
        self.down_conv_2=DownSample_2D(32,64)
        self.down_conv_3=DoubleConv(64,128)
        self.down_conv_4=DoubleConv(128,256)
        
        self.bottle_neck=DoubleConv(256,512)
        
        ## Define decoder convolution blocks
        self.up_conv_1=UpSample_2D(512,256,upscale_dim=False)
        self.up_conv_2=UpSample_2D(256,128,upscale_dim=False)
        self.up_conv_3=UpSample_2D(128,64)
        self.up_conv_4=UpSample_2D(64,32)
        
        ## Output scales channel dimension to num_classes
        self.out=nn.Conv2d(32,num_classes,kernel_size=1)
        
    def forward(self,x):
        down_1,p1=self.down_conv_1(x)
        down_2,p2=self.down_conv_2(p1)
        p3=self.down_conv_3(p2)
        p4=self.down_conv_4(p3)
        
        b=self.bottle_neck(p4)
        
        up_1=self.up_conv_1(b,p4)
        up_2=self.up_conv_2(up_1,p3)
        up_3=self.up_conv_3(up_2,down_2)
        up_4=self.up_conv_4(up_3,down_1)
        
        return self.out(up_4)

class Model_2DUNET(nn.Module):
    def __init__(self,in_channels,num_classes,U_layers=1):
        super(Model_2DUNET,self).__init__()
        self.Ulayers=nn.ModuleList()
        
        ## Define the Unet layers
        for i in range(U_layers):
            if i!=U_layers-1:
                self.Ulayers.append(UNet_2D(in_channels,in_channels))
            else:
                self.Ulayers.append(UNet_2D(in_channels,num_classes))
    
    def forward(self,x):
        x=torch.transpose(x,1,3)
        for i in range(len(self.Ulayers)):
            if i != len(self.Ulayers)-1:
                # print(x.shape)
                x=self.Ulayers[i](x)+x ## Skip connections between layers
                # print(x.shape)
            else:
                # print(x.shape)
                x=self.Ulayers[i](x)
                # print(x.shape)
        x=torch.transpose(x,1,3)
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
y_w=np.load("D:/O4W/Simulations/PINO/FNO_NETmix/Re300/data_grid_w_ext.npy")
x_w=np.load("D:/O4W/Simulations/PINO/FNO_SS/CFD/data_grid_w_ext_DiscretizationReduction0.99.npy")

# ## Shift data
# t_sim=2
# t_shift=2
# dt=0.1
# x_w,y_w=shift_data(data_grid_w[:5000,...],t_sim,t_shift,dt)

## Recover grid
grid_x, grid_y=recover_grid(x_w.shape[0],x_w.shape[1])


## Convert arrays into PyTorch tensor
x_w,y_w=torch.from_numpy(x_w),torch.from_numpy(y_w)
grid_x,grid_y=torch.from_numpy(grid_x),torch.from_numpy(grid_y)
values_mask=torch.from_numpy(values_mask)

## Create dataset

batch_size=5
ntrain=int(0.9*x_w.shape[0])
data_x=torch.stack([x_w,grid_x,grid_y],dim=-1)
data_y=torch.stack([y_w],dim=-1)

## Create data loaders
train_loader=DataLoader(torch.utils.data.TensorDataset(data_x[:ntrain,...], data_y[:ntrain,...]),batch_size=batch_size,shuffle=True)
test_loader=DataLoader(torch.utils.data.TensorDataset(data_x[ntrain:,...], data_y[ntrain:,...]),batch_size=batch_size,shuffle=False)


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
f_dim=data_x.shape[-1]
model=Model_2DUNET(f_dim,num_classes=1,U_layers=3).to(device)

## Define optimizer
learning_rate=(2.5/2**0)*1e-6
# step_size=int(6*32/5)
milestones=[5,15,25,35,45]
gamma=0.5
optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

## Define loss function
loss_fn=LpLoss(size_average=True)

#Load checkpoint
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
        
save_checkpoint("UNET2D_CFD_SS_Re300_5000p_w_DR0.99_epoch70", model, optimizer)
save_testdata(all_avg_train_data_loss,all_std_train_data_loss,all_avg_test_data_loss,all_std_test_data_loss,"UNET2D_CFD_SS_Re300_5000p_w_DR0.99_epoch70_testdata")

tstep=420
model_prediction=predict(model,data_x[tstep:(tstep+1),:,:,:])
visualise_data(model_prediction,data_x,0,tstep,"w")
visualise_test_error_plot(all_avg_test_data_loss,epoch)
