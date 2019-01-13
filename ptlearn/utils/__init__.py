import torch


def set_device(device, verbose=True):
    if isinstance(device, torch.device):
        use_device = device
    else:
        if device.find('cuda') == 0 and torch.cuda.is_available():
            use_device = torch.device(device)
        else:
            use_device = torch.device('cpu')
    if verbose:
        print('Use device: {}'.format(use_device))
    return use_device
