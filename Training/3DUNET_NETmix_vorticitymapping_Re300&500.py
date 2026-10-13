import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

from losses import LpLoss

import time
from timeit import default_timer
from functools import reduce
import operator


class DoubleConv_3D(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        
        ## Define convolutions which maintain the x, y and t dimensions
        self.conv_op=nn.Sequential(
            nn.Conv3d(in_channels,out_channels//2,kernel_size=3,padding=1),
            nn.BatchNorm3d(num_features=out_channels//2),
            nn.GELU(),
            nn.Conv3d(out_channels//2,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm3d(num_features=out_channels),
            nn.GELU()
            )
    def forward(self,x):
        return self.conv_op(x)
    

class DownSample_3D(nn.Module):
    def __init__(self, in_channels, out_channels,pooling=True):
        super().__init__()
        self.conv=DoubleConv_3D(in_channels,out_channels)
        
        ## Define max pooling which reduces x, y and t dimensions by 2
        self.pool=nn.MaxPool3d(kernel_size=2,stride=2)
        self.pooling=pooling
    
    def forward(self,x):
        down=self.conv(x)
        
        if self.pooling:
            p=self.pool(down)
        
            return down,p
        else:
            return down
    
class UpSample_3D(nn.Module):
    def __init__(self,in_channels,out_channels,upscale_dim=True):
        super().__init__()
        
        if upscale_dim:
            self.up=nn.ConvTranspose3d(in_channels,in_channels//2,kernel_size=2,stride=2)
        else:
            self.up=nn.Sequential(
                nn.Conv3d(in_channels,in_channels//2,kernel_size=3,padding=1),
                nn.BatchNorm3d(num_features=in_channels//2),
                nn.GELU(),
                )
        
        self.conv=DoubleConv_3D(in_channels,out_channels)
        
    def forward(self,x1,x2):
        x1=self.up(x1)
        
        ## Define Concatenation to enable skip connections
        x=torch.cat([x1,x2],axis=1)
        
        return self.conv(x)
    
    
class UNet_3D(nn.Module):
    def __init__(self, in_channels,num_classes,bottleneck_channel=512,level_channels=[64,128,256]):
        super().__init__()
        
        ## Define coder convolution block
        self.down_conv_1=DownSample_3D(in_channels=in_channels,out_channels=level_channels[0])
        self.down_conv_2=DownSample_3D(in_channels=level_channels[0],out_channels=level_channels[1])
        self.down_conv_3=DownSample_3D(in_channels=level_channels[1],out_channels=level_channels[2],pooling=False)
        
        self.bottle_neck=DoubleConv_3D(in_channels=level_channels[2],out_channels=bottleneck_channel)
        
        ## Define decoder convolution block
        self.up_conv_1=UpSample_3D(in_channels=bottleneck_channel,out_channels=level_channels[2],upscale_dim=False)
        self.up_conv_2=UpSample_3D(in_channels=level_channels[2],out_channels=level_channels[1])
        self.up_conv_3=UpSample_3D(in_channels=level_channels[1],out_channels=level_channels[0])
        
        ## Define output layer
        self.out=nn.Conv3d(in_channels=level_channels[0],out_channels=num_classes,kernel_size=1)
        
    def forward(self,x):
        down_1,p1=self.down_conv_1(x)
        down_2,p2=self.down_conv_2(p1)
        p3=self.down_conv_3(p2)
        
        b=self.bottle_neck(p3)
        
        up_1=self.up_conv_1(b,p3)
        up_2=self.up_conv_2(up_1,down_2)
        up_3=self.up_conv_3(up_2,down_1)
        
        return self.out(up_3)
        



class Model_3DUNET(nn.Module):
    def __init__(self,in_channels,num_classes,U_layers=1):
        super(Model_3DUNET,self).__init__()
        
        self.Ulayers=nn.ModuleList()
        
        ## Define the Unet layers
        for i in range(U_layers):
            if i!=U_layers-1:
                self.Ulayers.append(UNet_3D(in_channels,in_channels))
            else:
                self.Ulayers.append(UNet_3D(in_channels,num_classes))
    
    def forward(self,x):
        batch_size,nx,ny,nt,f=x.shape[0],x.shape[1],x.shape[2],x.shape[3],x.shape[4]
        # print(x.shape)
        x=F.pad(x,(0,0,1,1),"constant",0)
        x=torch.permute(x,(0,4,3,1,2))
        # print(x.shape)
        for i in range(len(self.Ulayers)):
            if i!=len(self.Ulayers)-1:
                x=self.Ulayers[i](x)+x ## Skip connections between layers
            else:
                x=self.Ulayers[i](x)
            
        x=torch.permute(x,(0,3,4,2,1))
        
        return x[...,:nt,:]



def compute_data_loss(s_pred, s_true,mask,use_sum=True):
    mask=mask.to(device, dtype=torch.float32)
    #print(s_pred[...,0].shape)
    p_pred = torch.mul(s_pred[...,0],mask)
    #print(p_pred.shape)
    p_true = torch.mul(s_true[...,0],mask)
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



def recover_grid_3D(n_space,res_space,res_time,group,dist,t_sim):
    def expand_space(data,n_space,group):
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
    
def recover_grid_2D(n_space,res_space,res_time,group,dist,t_sim):
    
    def expand_space(data,n_space):
        data=np.repeat(data[np.newaxis,...],n_space,axis=0)
        return data
    
    def expand_time(data,group,dist):
        aux=np.zeros(((data.shape[0]-dist*(group-1)),group))
        for n in range(aux.shape[0]):
            aux[n,:]=data[n:(n+dist*(group-1)+1):dist]
        return aux
         
    
    x=np.linspace(0,1,res_space)
    y=np.linspace(0,1,res_space)
    t=np.linspace(0,t_sim,res_time)
    
    x_grid,y_grid=np.meshgrid(x,y)
    
    x_grid=expand_space(x_grid.copy(),n_space)
    y_grid=expand_space(y_grid.copy(),n_space)
    t_grid=expand_time(t.copy(),group,dist)
    
    return x_grid, y_grid, t_grid


def group_data_2D(data,group,dist):
    
    def produce_x(data,group,dist):
        return data[:(data.shape[0]-dist*(group-1)),...]
    
    def produce_y(data,group,dist):
        aux=np.zeros(((data.shape[0]-dist*(group-1)),data.shape[1],data.shape[2],group))
        for n in range(aux.shape[0]):
            aux[n,:,:,:]=np.transpose(data[n:(n+dist*(group-1)+1):dist,...],(1,2,0))
        return aux
        
    x=produce_x(data,group,dist)
    y=produce_y(data,group,dist)
    
    return x,y
  
def group_data_3D(data,group,dist):
    
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

def predict(model,x):
    model.eval()
    # print(x_b.shape,x_t.shape)
    with torch.no_grad():
        x=x.to(device, dtype=torch.float32)
    return model(x)

def visualise_data(NN_pred,y_arr,tpred,group,tstep):
    NN_pred=NN_pred.cpu().detach().numpy()
    y_arr=y_arr.cpu().detach().numpy()
    plt.figure(num=1,figsize=(8, 8))
    plt.subplot(2,1,1)
    plt.contourf(NN_pred[tpred,:,:,group-1,0], extent=[0, 1, 0, 1], origin='lower', cmap='coolwarm')
    plt.colorbar(label="predicted")
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.grid(True)
    plt.axis('equal')

    plt.subplot(2,1,2)
    plt.contourf(y_arr[tstep+tpred,:,:,group-1,0], extent=[0, 1, 0, 1], origin='lower', cmap='coolwarm')
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

    
def load_checkpoint(model,optimizer,path,device):
    ckpt=torch.load(path,map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print("Model and optimizer sucessfully loaded!")
    
def save_testdata(train_avg,train_std,test_avg,test_std,file_name):
    to_save=np.stack((train_avg,train_std,test_avg,test_std),axis=-1)
    np.savetxt(file_name+".csv",to_save,delimiter=",")
    print("Test data was saved!")

# def save_testdata(data,file_name):
#     np.savetxt(file_name+".csv",data,delimiter=",")
#     print("Test data was saved!")

def count_params(model):
    c=0
    for p in model.parameters():
        c+=reduce(operator.mul,list(p.size()))
    return c    
    
    
## Loading data
values_mask=np.load("D:/O4W/Simulations/PINO/FNO_NETmix/Re500/values_mask_Re300&Re500.npy")
data_x_train=np.load("D:/O4W/Simulations/PINO/FNO_NETmix/Re500/data_x_train_Re300&Re500.npy")
data_x_test=np.load("D:/O4W/Simulations/PINO/FNO_NETmix/Re500/data_x_test_Re300&Re500.npy")
data_y_train=np.load("D:/O4W/Simulations/PINO/FNO_NETmix/Re500/data_y_train_Re300&Re500.npy")
data_y_test=np.load("D:/O4W/Simulations/PINO/FNO_NETmix/Re500/data_y_test_Re300&Re500.npy")

## Group data
group=26
dist=20


## Convert arrays into PyTorch tensor
data_x_train=torch.from_numpy(data_x_train)
data_x_test=torch.from_numpy(data_x_test)
data_y_train=torch.from_numpy(data_y_train)
data_y_test=torch.from_numpy(data_y_test)
values_mask=torch.from_numpy(values_mask)

## Create data loader
batch_size=5
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

# device=torch.device("cpu")

## Define model
f_dim=4
model=Model_3DUNET(in_channels=f_dim, num_classes=1,U_layers=2).to(device)  

## Define optimizer
learning_rate=(5/2**5)*1e-6
milestones=[5]
gamma=0.5
optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)


## Define loss function
loss_fn=LpLoss(size_average=True)

## Load checkpoint
load_checkpoint(model,optimizer,"3DUNET_Re300&500_5000p_w_PH2e-1_epoch65.pth", device)


## Training loop
epoch=5
all_avg_train_data_loss=[]
all_std_train_data_loss=[]
all_avg_test_data_loss=[]
all_std_test_data_loss=[]
start_time=time.time()
#padding=1
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
        data_loss=compute_data_loss(out,y,mask=values_mask[:X.shape[0],...],use_sum=True) # Compute the loss
        #optimizer.zero_grads() # Zero gradients
        data_loss.backward() # Backpropagate
        optimizer.step() # Update weights
        #scheduler.step()
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
        for (X,y) in test_loader:
            X=X.to(device, dtype=torch.float32) # Move input data to the device
            y=y.to(device,dtype=torch.float32)
            out=model(X)
            data_loss=compute_data_loss(out,y,mask=values_mask[:X.shape[0],...],use_sum=True).item()
            test_data_loss.append(data_loss)
        avg_test_data_loss=np.mean(test_data_loss)
        std_test_data_loss=np.std(test_data_loss)
                
        all_avg_test_data_loss.append(avg_test_data_loss.copy())
        all_std_test_data_loss.append(std_test_data_loss.copy())
        t2=default_timer()
             
        # scheduler.step()
    print(f"Test average data loss: {avg_test_data_loss:>8f},\nTest std data loss: {std_test_data_loss:>8f},\nTrain average data loss: {avg_train_data_loss:>8f},\nTrain std data loss: {std_train_data_loss:>8f},\nTime (s): {t2-t1:.3f} \n")
print("Training complete!")
elapsed_time=time.time()-start_time
print(f"Training time (s): {elapsed_time:.3f}\n")
        
save_checkpoint("3DUNET_Re300&500_5000p_w_PH2e-1_epoch70", model, optimizer)
save_testdata(all_avg_train_data_loss,all_std_train_data_loss,all_avg_test_data_loss,all_std_test_data_loss,"3DUNET_Re300&500_5000p_w_PH2e-1_epoch70_testdata")
# save_testdata(all_avg_test_data_loss,"3DUNET_Re300&500_5000p_w_PH2e-1_epoch05_testdata")

tstep=215
model_prediction=predict(model,data_x_test[tstep:(tstep+2),...])
visualise_data(model_prediction,data_y_test,0,5,tstep)
visualise_test_error_plot(all_avg_test_data_loss,epoch)        