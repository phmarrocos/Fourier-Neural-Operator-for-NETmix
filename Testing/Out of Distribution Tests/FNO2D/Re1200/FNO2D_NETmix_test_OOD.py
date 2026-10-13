import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from basics import SpectralConv2d
import numpy as np
import matplotlib.pyplot as plt
from losses import LpLoss

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
    p_true = torch.mul(s_true[..., 0],mask)

        
    lploss = LpLoss(size_average=True)
    loss_p = lploss(p_pred, p_true)
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
    #print(x.shape)
    with torch.no_grad():
        x=x.to(device, dtype=torch.float32)
        #batch_size, S, S, T, nfields = x.shape
        #x_in=F.pad(x, (0, 0, 0, padding), "constant", 0)
        #out=model(x_in).reshape(batch_size, S, S, T + padding, nfields)
         #print(pred)
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
    
def load_checkpoint(model,optimizer,path,device):
    ckpt=torch.load(path,map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    print("Model and optimizer sucessfully loaded!")
    
with torch.no_grad():
    torch.cuda.empty_cache()
    
    
## Loading data
values_mask=np.load("/O4W/Simulations/PINO/NETmix_Data/Re1200/Co0p1/data_grid_mask.npy")
data_grid_w=np.load("/O4W/Simulations/PINO/NETmix_Data/Re1200/Co0p1/data_grid_w_ext.npy")

## Shift data
t_sim=2
t_shift=1
dt=2/5000
x_w,y_w=shift_data(data_grid_w[:5000,...],t_sim,t_shift,dt)

## Recover grid
grid_x, grid_y=recover_grid(x_w.shape[0],data_grid_w.shape[1])

## Random sampling of the data
num_cases=2500
idx=np.random.choice(range(x_w.shape[0]),size=num_cases,replace=False)
grid_x=grid_x[idx,:,:]
grid_y=grid_y[idx,:,:]
x_w=x_w[idx,:,:]
y_w=y_w[idx,:,:]


## Convert arrays into PyTorch tensor
x_w,y_w=torch.from_numpy(x_w),torch.from_numpy(y_w)
grid_x,grid_y=torch.from_numpy(grid_x),torch.from_numpy(grid_y)
values_mask=torch.from_numpy(values_mask)

## Create dataset
data_x=torch.stack([x_w[...,0],grid_x,grid_y],dim=-1)
data_y=torch.stack([y_w],dim=-1)

## Create test loader
test_loader=DataLoader(torch.utils.data.TensorDataset(data_x, data_y),batch_size=1,shuffle=True)

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
model= FNN2d(modes1=[16,16,16,16],
              modes2=[16,16,16,16],
              fc_dim=128,
              layers=[64,64,64,64],
              in_dim=3,
              out_dim=1,
              activation="gelu"  
        ).to(device)

## Define loss function
loss_fn=LpLoss(size_average=True)

# ## Load checkpoint
load_checkpoint(model,None,"/O4W/Simulations/PINO/FNO_NETmix/Re300/FNO2D_Re300_5000p_w_dt4e-4_1inlet_epoch70.pth", device)

## Test the model

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
            
print(f"Test average data loss: {avg_test_data_loss:>8f} \n")

tstep=101
model_prediction=predict(model,data_x[tstep:(tstep+1),:,:,:])
visualise_data(model_prediction,data_y,0,tstep,"w")
