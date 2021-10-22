import numpy as np
import pandas as pd
import datetime
import time
from model import Net1
import torch.optim as optim
from scipy import io
import argparse
import os                    # nn.BatchNorm2d(2,affine=False),
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import h5py  
import matplotlib.pyplot as plt
import h5py  
import matplotlib
from PIL import Image
import math
from sklearn.metrics import confusion_matrix
import pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import itertools
import ParaSetting

torch.cuda.synchronize()
starttime = time.time()

os.environ["CUDA_VISIBLE_DEVICES"]="0" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
read_h5 = True
num_workers = 0


## Working path define
expname  = ParaSetting.expname
root_ws   = ParaSetting.root_ws

epoch_num_char = str(ParaSetting.epoch_num)

datapath  = root_ws+expname+'data/'
modelpath_test = root_ws+expname+'model/'+'epoch-'+epoch_num_char+'.pth'
savepath  = root_ws+expname+'results/'
os.makedirs(savepath, exist_ok=True)

## Hyperparameters
Nx = ParaSetting.Nx
bs = ParaSetting.bs  # batch size

class prepareData_test(Dataset):
    def __init__(self, train_or_test):

       self.files = os.listdir(datapath+train_or_test)
       self.files.sort(key=lambda x:int(x[:-4])) 
       self.train_or_test= train_or_test

    def __len__(self):
       return len(self.files)

    def __getitem__(self, idx):
        
        data = torch.load(datapath+self.train_or_test+'/'+self.files[idx])
        return data['k-space'],  data['label']
   

testset = prepareData_test('test')
testloader = torch.utils.data.DataLoader(testset, batch_size=1,shuffle=False, num_workers=num_workers)


# Testing
filename = os.listdir(datapath+'/test/')
length = len(filename)

din = np.empty((length,2,Nx,10))
dout = np.empty((length,2,Nx,1))
dlab = np.empty((length,2,Nx,1))

model = torch.load(modelpath_test)
criterion1 = nn.MSELoss()

model.eval()
loss_validation_list = []
loss_batch = []
loss = []
data1 = []
print('\n testing...')
for i, data in enumerate(testloader, 0):
    inputs = data[0].reshape(-1,2,Nx,10).to(device)
    labels = data[1].reshape(-1,2,Nx,1).to(device)
   
    with torch.no_grad():
         outs = model(inputs)
    
    loss = criterion1(outs, labels)
    loss_batch.append(loss.item())
    loss_validation_list.append(round(sum(loss_batch) / len(loss_batch),10))
    print(loss_validation_list)
    output = outs.cuda().data.cpu()
    labelo =labels.cuda().data.cpu()
    inputo = inputs.cuda().data.cpu()
    
    dout[i] = output[0:1,:,:,:]
    dlab[i] = labelo[0:1,:,:,:]
    
f = h5py.File(savepath+'output.h5','w')
f['k-space'] = dout
k = h5py.File(savepath+'label.h5','w')
k['k-space'] = dlab
f.close()
k.close()

ksp = dlab-dout
ksp = ksp[:,0,:,:]+1j*ksp[:,1,:,:]
ksp = ksp.transpose(1,0,2)
ksp = np.reshape(ksp, (128, 126, 32,2), order="F")
ksp = ksp[:,:,:,0]-ksp[:,:,:,1]

plt.figure(1)
plt.imshow(np.log10(abs(ksp[:,:,15])),cmap='gray')
plt.savefig(savepath+'ksp.tif')