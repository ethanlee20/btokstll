
import torch.cuda


def print_gpu_memory_summary():
    print(torch.cuda.memory_summary(abbreviated=True))


def gpu_peak_memory_usage():
    return f"{torch.cuda.max_memory_allocated()/1024**3:.5f} GB"


def select_device():
    """
    Select a device to compute with.

    Returns
    -------
    str
        The name of the selected device.
        "cuda" if cuda is available,
        otherwise "cpu".
    """

    device = (
        "cuda" 
        if torch.cuda.is_available()
        else 
        "cpu"
    )
    print("Device: ", device)
    return device