import os

import numpy as np
import h5py
import torch
from torchvision.utils import save_image
import onnxruntime

import maths_new
import utils.helpers


CONFIG_PATH = './configs'


def main(args):
    config = utils.helpers.load_config(CONFIG_PATH, args.config)
    data_path = os.path.normpath(config['paths']['data_path'])
    output_path = os.path.normpath(config['paths']['output_path'])
    model_path = os.path.normpath(config['paths']['model_path'])

    os.makedirs(os.path.join(output_path), exist_ok=True)

    for file in os.listdir(data_path):
        if ('_input.h5') in file:
            contrast = file.split('/')[-1].split('_')[-2]
            h5 = h5py.File(os.path.join(data_path, file), 'r')
            data = h5['img'][()] 
            input = torch.from_numpy(data.copy())
            data = np.flip(data,axis=3).transpose(1,2,3,0)
            data = torch.from_numpy(data.copy())
            
            if contrast == 'T1w':
                if file.split('/')[-1].split('_')[0] == 'synthetic':
                    norm1, norm2 = 2.2, 1.5
                else:
                    data = maths_new.minmax_pc_norm_tc(data, (0.0, 0.94))   
                    norm1, norm2 = 3.5, 2.2
            else:
                data = maths_new.minmax_norm_tc(data, (0.0, 1.0)) 
                norm1, norm2 = 1.8, 1.5

            data = data.unsqueeze(0).float()

            ### inference by compiled onnx model
            if (contrast == 'T1w'):
                model = os.path.join(model_path, 't1w_pfsr_model.onnx')
            else:
                model = os.path.join(model_path, 't2w_pfsr_model.onnx')

            ort_session = onnxruntime.InferenceSession(model)
            ort_inputs = {ort_session.get_inputs()[0].name: (data.numpy())}
            ort_outs = ort_session.run(None, ort_inputs)
            ort_outs = ort_outs[0]
            print(np.shape(ort_outs), type(ort_outs))

            c = torch.transpose(torch.from_numpy(ort_outs), 0, 4).flip(3) 
            c = torch.rot90(c,1,[-2, -3])
            c = c.squeeze()                  
            c = c.flip(2)
            c = c.unsqueeze(1)*1

            output_fname = file.split('/')[-1].split('.')[0].split('_')[0:3]
            output_fname = '_'.join(output_fname)
            save_image(
                c*norm1,
                os.path.join(output_path, output_fname + '_output.png'),
                normalize=False, value_range=(0, 1), nrow=8, padding=2
            )

            cc = torch.transpose(torch.from_numpy(ort_outs), 0, 4).flip(3)           
            with h5py.File(os.path.join(output_path, output_fname + '_output.h5'), 'w') as f:
                f.create_dataset('img', data=cc)
                f.close 

            input_fname = file.split('/')[-1].split('.')[0].split('_')[0:3]
            input_fname = '_'.join(input_fname)
            save_image(
                torch.rot90(input,1,[-1, -2]).flip(3) / input.max() * norm2,
                os.path.join(output_path, input_fname + '_input.png'),
                normalize=False, value_range=(0, 1), nrow=8, padding=2
            )


if __name__ == '__main__':
    default_fname = 'inference.yaml'
    args = utils.helpers.build_args(default_fname)
    main(args)