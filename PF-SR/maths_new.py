import numpy as np
import torch

def zscore_norm(data, eps: float = 1e-8, adjusted: bool = False):
    """Per-instance data z-score standardisation
    """
    if np.iscomplexobj(data):
        data.real = (data.real - np.mean(data.real)) / (np.std(data.real) + eps) # adjusted variance
        data.imag = (data.imag - np.mean(data.imag)) / (np.std(data.imag) + eps)
    else:
        data = (data - np.mean(data)) / (np.std(data) + eps)
    return data


def minmax_norm(data, new_drange: tuple = (0.0, 1.0)):
    """Per-instance data min-max normalisation
    """
    if isinstance(data, torch.Tensor):
        data_min, data_max = data.min(), data.max()
        data = (new_drange[1] - new_drange[0]) * (data - data_min) / (data_max - data_min) + new_drange[0]
        return data

    if np.iscomplexobj(data):
        re_min, re_max = np.min(data.real), np.max(data.real)
        im_min, im_max = np.min(data.imag), np.max(data.imag)

        data_re = (data.real - re_min) / (re_max - re_min)
        data_im = (data.imag - im_min) / (im_max - im_min)
        return data_re + 1j*data_im
    else:
        data_min, data_max = np.min(data), np.max(data)
        data = (new_drange[1] - new_drange[0]) * (data - data_min) / (data_max - data_min) + new_drange[0]
        return data


def zscore_norm_t(data: torch.Tensor, eps: float = 1e-8):
    """Per-instance data z-score standardisation
    """
    mean = data.mean()
    std = data.std()
    data = (data - mean) / (std + eps)
    return data, mean, std


def minmax_norm_t(data: torch.Tensor, new_drange: tuple = (0.0, 1.0)):
    """Per-instance data min-max normalisation
    """
    data_min, data_max = data.min(), data.max()
    data = (new_drange[1] - new_drange[0]) * (data - data_min) / (data_max - data_min) + new_drange[0]
    return data


def zscore_norm_tc(data: torch.Tensor, eps: float = 1e-8):
    """Per-instance, per-channel data z-score standardisation
    """
    mean = data.mean(dim=(-1, -2, -3), keepdim=True)
    std = data.std(dim=(-1, -2, -3), keepdim=True)
    data = (data - mean) / (std + eps)
    return data, mean, std


def minmax_norm_tc(data: torch.Tensor, new_drange: tuple = (-0.05, 1.05)):
    """Per-instance, per-channel data min-max normalisation
    """
    data_min, data_max = data.amin(dim=(-1, -2, -3), keepdim=True), data.amax(dim=(-1, -2, -3), keepdim=True)
    data = (new_drange[1] - new_drange[0]) * (data - data_min) / (data_max - data_min) + new_drange[0]
    return data

def minmax_per_norm_tc(data: torch.Tensor,percentile:tuple = (0.0, 0.998), new_drange: tuple = (0.0, 1.0)):
    q = torch.tensor([percentile[0], percentile[1]], dtype=torch.float64)
    data_p = torch.quantile(data.flatten(), q)
    data = (new_drange[1] - new_drange[0]) * (data - data_p[0]) / (data_p[1] - data_p[0]) + new_drange[0]
    return data


def minmax_per_norm_tc_t1(data: torch.Tensor, percentile: tuple = (0.01, 0.97), new_drange: tuple = (0.0, 1.0)):
    """Per-instance, per-channel data min-max normalisation
    """
    q = torch.tensor([percentile[0], percentile[1]], dtype=torch.float64)
    # data_min, data_max = data.amin(dim=(-1, -2, -3), keepdim=True), data.amax(dim=(-1, -2, -3), keepdim=True)
    # data = (new_drange[1] - new_drange[0]) * (data - data_min) / (data_max - data_min) + new_drange[0]

    for i in range(0, data.shape[0]):
        data_p = torch.quantile(data[i].flatten(), q)
        data[i].clamp_(data_p[0], data_p[1])
    return data

def minmax_pc_norm_tc(data: torch.Tensor, percentile: tuple = (0.0, 0.95), new_drange: tuple = (0.0, 1.0)):
    """Per-instance, per-channel data min-max normalisation
    """
    q = torch.tensor([percentile[0], percentile[1]], dtype=torch.float32)
    data_min, data_max = data.amin(dim=(-1, -2, -3), keepdim=True), data.amax(dim=(-1, -2, -3), keepdim=True)
    data = (new_drange[1] - new_drange[0]) * (data - data_min) / (data_max - data_min) + new_drange[0]

    for i in range(0, data.shape[0]):
        data_p = torch.quantile(data[i].flatten(), q)
        data[i].clamp_(data_p[0], data_p[1])
    return data

def fft3c(data):
    """
    Apply centred 3 dimensional Fast Fourier Transform.
    Args:
        data (np.array): Complex valued input data containing at least 3
            dimensions: dimensions -3, -2 & -1 are spatial dimensions.
            All other dimensions are assumed to be batch dimensions.
    Returns:
        np.array: The FFT of the input.
    """
    data = np.fft.ifftshift(data, axes=(-1, -2, -3))
    data = np.fft.fftn(data, axes=(-1, -2, -3), norm='ortho')
    data = np.fft.fftshift(data, axes=(-1, -2, -3))
    return data


def ifft3c(data):
    """
    Apply centred 3 dimensional inverse Fast Fourier Transform.
    Args:
        data (np.array): Complex valued input data containing at least 3
            dimensions: dimensions -3, -2 & -1 are spatial dimensions.
            All other dimensions are assumed to be batch dimensions.
    Returns:
        np.array: The FFT of the input.
    """
    data = np.fft.fftshift(data, axes=(-1, -2, -3))
    data = np.fft.ifftn(data, axes=(-1, -2, -3), norm='ortho')
    data = np.fft.ifftshift(data, axes=(-1, -2, -3))
    return data