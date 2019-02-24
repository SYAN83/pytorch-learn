import torch


def set_device(device, verbose=True):
    cuda_available = torch.cuda.is_available()
    if isinstance(device, torch.device):
        current_device = device
    elif isinstance(device, str):
        if device.find('cuda') > -1 and cuda_available:
            current_device = torch.device(device)
        else:
            current_device = torch.device('cpu')
    if verbose:
        print('CUDA is currently available:', cuda_available)
        if cuda_available:
            print('The number of GPUs available:', torch.cuda.device_count())
        print('Current device:', current_device)
    return current_device
