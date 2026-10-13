# -*- coding: utf-8 -*-
import numpy as np

## Select points for Re=300
hf_Re300=np.load("D:/O4W/Simulations/PINO/FNO_NETmix/Re300/data_grid_w_ext.npy")
lf_Re300=np.load("D:/O4W/Simulations/PINO/FNO_SS/CFD/data_grid_w_ext_DiscretizationReduction0.9.npy")
idx_Re300=np.random.choice(range(hf_Re300.shape[0]),size=int(hf_Re300.shape[0]/2),replace=False)
ntrain=int(hf_Re300.shape[0]*0.9/2)
hf_Re300=hf_Re300[idx_Re300,...]
hf_Re300_train=hf_Re300[:ntrain,...]
hf_Re300_test=hf_Re300[ntrain:,...]
lf_Re300=lf_Re300[idx_Re300,...]
lf_Re300_train=lf_Re300[:ntrain,...]
lf_Re300_test=lf_Re300[ntrain:,...]

del hf_Re300,lf_Re300

## Select points for Re=500
hf_Re500=np.load("D:/O4W/Simulations/PINO/NETmix_Data/Re500/Co0p1/data_grid_w_ext.npy")
lf_Re500=np.load("D:/O4W/Simulations/PINO/FNO_SS/CFD/data_grid_w_ext_DiscretizationReduction0.9_Re500.npy")
idx_Re500=np.random.choice(range(hf_Re500.shape[0]),size=int(hf_Re500.shape[0]/2),replace=False)
hf_Re500=hf_Re500[idx_Re500,...]
lf_Re500=lf_Re500[idx_Re500,...]
hf_Re500_train=hf_Re500[:ntrain,...]
hf_Re500_test=hf_Re500[ntrain:,...]
lf_Re500_train=lf_Re500[:ntrain,...]
lf_Re500_test=lf_Re500[ntrain:,...]

del hf_Re500, lf_Re500

data_lf_train=np.concatenate((lf_Re300_train,lf_Re500_train),axis=0)
data_hf_train=np.concatenate((hf_Re300_train,hf_Re500_train),axis=0)
data_lf_test=np.concatenate((lf_Re300_test,lf_Re500_test),axis=0)
data_hf_test=np.concatenate((hf_Re300_test,hf_Re500_test),axis=0)       


np.save("data_lf_train_Re300&Re500",data_lf_train)
np.save("data_hf_train_Re300&Re500",data_hf_train)
np.save("data_lf_test_Re300&Re500",data_lf_test)
np.save("data_hf_test_Re300&Re500",data_hf_test)



