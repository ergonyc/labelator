import torch


def get_usable_device(device):
    """
    gets the usable device
    """

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    elif device == "cuda":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "mps":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = "cpu"
    return device
