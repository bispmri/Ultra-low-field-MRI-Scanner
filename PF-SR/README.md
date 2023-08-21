# PF-SR

This repository contains an implementation of PF-SR, a 3D deep learning model for reduction of PF-related artifacts, and noise, and increase in spatial resolution of ULF MRI brain data.

Architecture of the 3D PF-SR deep learning reconstruction model



## Data preparation

Preparation of training data can be found in prepare_train_data.py, which processes the raw HCP data and generates the training input and target. It includes the image downsampling through local mean to generate the training target, symmetric k-space truncation in three directions, 2D PF sampling of a fraction of 0.7 along two PE directions (left-right and superior-inferior), and addition of Rician noise in magnitude image to generate the training input. The access of T1w and T2w HCP data can be requested at https://www.humanconnectome.org/study/hcp-young-adult/document/1200-subjects-data-release.

## Testing

Test on synthetic and experimental T1w and T2w ULF data can be found in test.py, where low-resolution (3mm isotropic), noisy, 2D PF-sampled 3D data is fed to PF-SR model, which reduces artifacts and noise, and improves spatial resolution to synthetic 1.5mm isotropic resolution.

## Suggested setup

The codes were implemented on Python 3.9.0 installed on Ubuntu 18.04.5 LTS. More details can be found in dependencies.txt.

## Update notes

8/2023: initial release v1

This project is continuously developing, stay tuned for future developments and new data and/or model release.

For more details, go visit the related manuscript

- V. Lau, L. Xiao, Y. Zhao, S. Su, Y. Ding, C. Man, X. Wang, A. Tsang, P. Cao, G. K. K. Lau, G. K. K. Leung, A. T. L. Leong, E. X. Wu, Pushing the limits of low-cost ultra-low-field MRI by dual-acquisition deep learning 3D superresolution. *Magn Reson Med* 90, 400-416 (2023).
