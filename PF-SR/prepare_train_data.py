import glob
import os
import shutil

import numpy as np
import scipy, scipy.io
import joblib

import h5py
import monai

import nibabel as nib
import skimage.transform, skimage.io
import sklearn.model_selection

import maths_new
import utils.helpers


CONFIG_PATH = './configs'


def centre_crop(data, shape):
    d_from = (data.shape[-3] - shape[0]) // 2
    w_from = (data.shape[-2] - shape[1]) // 2
    h_from = (data.shape[-1] - shape[2]) // 2
    d_to = d_from + shape[0]
    w_to = w_from + shape[1]
    h_to = h_from + shape[2]
    return data[d_from:d_to, w_from:w_to, h_from:h_to]


def kspace_centre_crop(data, shape):
    data = maths_new.fft3c(data)
    data = centre_crop(data, shape)
    data = maths_new.ifft3c(data)
    data = np.abs(data)  
    return data


def kspace_pocs(data,ratio):
    data = maths_new.fft3c(data)
    size = data.shape 
    pf_line1 = int(size[0]*ratio)
    pf_line3 = int(size[2]*ratio)
    data[0:pf_line1, :, :] = 0
    data[:, :,-pf_line3::] = 0
    data = maths_new.ifft3c(data)
    data = np.abs(data)  
    return data


def create_h5(raw_fname, output_path, lr_image_size):
    uid = raw_fname.split('/')[-1].split('_')[0]
    contrast = raw_fname.split('/')[-1].split('_')[-2]
    image = nib.nifti1.load(raw_fname).get_fdata()
          
    image_hr_intra = skimage.transform.downscale_local_mean(image, (2, 2, 2))

    image_hr_intra = centre_crop(image_hr_intra,[128,160,144])
    image_hr = np.zeros([160, 160, 144])
    image_hr[16:-16, :, :] = image_hr_intra

    image_lr = kspace_centre_crop(image_hr, lr_image_size)
    image_lr = kspace_pocs(image_lr,0.3)
    noise = monai.transforms.RandRicianNoise(prob=1.0, std=0.4, relative=True, channel_wise=True, sample_std=False)
    noise_seed = np.random.randint(10000)
    noise.set_random_state(noise_seed)
    image_lr = noise(image_lr)
    
    new_fname = raw_fname.split('/')[-1].split('.')[0]

    os.makedirs(os.path.join(output_path, contrast), exist_ok=True)

    with h5py.File(os.path.join(output_path, contrast, new_fname+'.h5'), 'w') as f:
        f.create_dataset('data', data=image_lr)
        f.create_dataset('target', data=image_hr)
        f.attrs['uid'] = uid


def main(args):
    config = utils.helpers.load_config(CONFIG_PATH, args.config)
    data_path = os.path.normpath(config['paths']['data_path'])
    output_path = os.path.normpath(config['paths']['output_path'])

    seed = config['seed']

    train_test_ratio = config['train_test_ratio']
    validation_test_ratio = config['validation_test_ratio']

    lr_image_size = config['lr_image_size']


    print("Start processing data")
    # we use the original filename ****_T1w_MPRx.nii.gz from HCP S1200 here
    raw_flist = glob.glob(os.path.join(data_path, '*.nii.gz'))
    print("Total number of files is %d" %len(raw_flist))
    print("Unprocessed number of files is %d" %len(raw_flist))

    _ = joblib.Parallel(n_jobs=1, verbose=1)(
        joblib.delayed(create_h5)(raw_fname, output_path, lr_image_size) for raw_fname in raw_flist
    )

    for contrast in ['T1w', 'T2w']:
        contrast_path = os.path.join(output_path, contrast)

        data_flist = glob.glob(os.path.join(contrast_path, '*.h5'))

        rng = np.random.default_rng(seed)
        rng.shuffle(data_flist)
        train, _test = \
            sklearn.model_selection.train_test_split(                          
            data_flist, test_size=train_test_ratio, random_state=seed
        )
        test, val = \
            sklearn.model_selection.train_test_split(
            _test, test_size=validation_test_ratio, random_state=seed
        )

        os.makedirs(os.path.join(contrast_path, 'train'), exist_ok=True)
        for file in train:
            file = file.split('/')[-1]
            shutil.move(
                contrast_path + '/' + file,
                os.path.join(contrast_path, 'train') + '/' + file
            )

        os.makedirs(os.path.join(contrast_path, 'test'), exist_ok=True)
        for file in test:
            file = file.split('/')[-1]
            shutil.move(
                contrast_path + '/' + file,
                os.path.join(contrast_path, 'test') + '/' + file
            )

        os.makedirs(os.path.join(contrast_path, 'validation'), exist_ok=True)
        for file in val:
            file = file.split('/')[-1]
            shutil.move(
                contrast_path + '/' + file,
                os.path.join(contrast_path, 'validation') + '/' + file
            )


if __name__ == '__main__':
    default_fname = 'process_data.yaml'
    args = utils.helpers.build_args(default_fname)
    main(args)

