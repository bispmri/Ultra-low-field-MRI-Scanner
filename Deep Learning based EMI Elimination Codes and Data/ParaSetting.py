# .mat data are saved in     (root_ws+expname+'data/matdata/')
# .pth data will be saved in (root_ws+expname+'data/')
#  model will be saved in    (root_ws+expname+'model/')
#  results will be saved in  (root_ws+expname+'results/')

## Define working path 
expname  = 'demo/'
root_ws  = '~/'  # file for saving intermediate data, e.g., model and .pth data (on workstation)

CH = 1 # num of receiving coils

## Define parameters
Nx = 128
bs = 16
epoch_num = 20
lr = 5e-4
lr_update = 0.9




