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

## Working path define
expname  = ParaSetting.expname
root_ws   = ParaSetting.root_ws

epoch_num_char = str(ParaSetting.epoch_num)

datapath  = root_ws+expname+'data/'
modelpath = root_ws+expname+'model/'

os.makedirs(modelpath, exist_ok=True)
os.makedirs(savepath, exist_ok=True)

## Hyperparameters
epoch_num = ParaSetting.epoch_num #iteration number 
Nx = ParaSetting.Nx
bs = ParaSetting.bs  # batch size

lr = ParaSetting.lr
lr_update = ParaSetting.lr_update
weight_decay = 0.000


class prepareData_train(Dataset):
    def __init__(self, train_or_test):

       self.files = os.listdir(datapath+train_or_test)
       self.train_or_test= train_or_test
    def __len__(self):
       return len(self.files)

    def __getitem__(self, idx):
        
        data = torch.load(datapath+self.train_or_test+'/'+self.files[idx])
        return data['k-space'],  data['label']

    
trainset = prepareData_train('train')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs,shuffle=True, num_workers=num_workers)

validationset = prepareData_train('validation')
validationloader = torch.utils.data.DataLoader(validationset, batch_size=bs,shuffle=True, num_workers=num_workers)

model = Net1().to(device)
print(model)

criterion1 = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


loss_train_list = []
loss_validation_list = []

for epoch in range(epoch_num):   
    model.train()
    loss_batch = []
    for i, data in enumerate(trainloader, 0):
       
        inputs = data[0].reshape(-1,2,Nx,10).to(device)
        labels = data[1].reshape(-1,2,Nx,1).to(device)

        outs = model(inputs)
        
        loss = criterion1(outs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_batch.append(loss.item())  
        if (i)%20==0:
            print('epoch:%d - %d, loss:%.10f'%(epoch+1,i+1,loss.item()))
    
    loss_train_list.append(round(sum(loss_batch) / len(loss_batch),10))
    print(loss_train_list)

    model.eval()     # evaluation
    
    loss_batch = []
    print('\n testing...')
    for i, data in enumerate(validationloader, 0):
        
        inputs = data[0].reshape(-1,2,Nx,10).to(device)
        labels = data[1].reshape(-1,2,Nx,1).to(device)
    
        with torch.no_grad():
            outs = model(inputs)
        loss = criterion1(outs, labels)
        loss_batch.append(loss.item())
        

    loss_validation_list.append(round(sum(loss_batch) / len(loss_batch),10))
    print(loss_validation_list)

    if (epoch+1) == epoch_num:
        torch.save(model, os.path.join(modelpath, 'epoch-%d.pth' % (epoch+1)))

   
    if (epoch+1) % 4 == 0:
        lr = min(2e-5,lr*lr_update) 
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

torch.cuda.synchronize()
endtime = time.time()
print('Finished Training. Training time elapsed %.2fs.' %(endtime-starttime))

