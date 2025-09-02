
import torch


def select_device():

    """
    Select a device to compute with.

    Returns the name of the selected device.
    "cuda" if cuda is available, otherwise "cpu".
    """
    
    device = (
        "cuda" 
        if torch.cuda.is_available()
        else "cpu"
    )
    print("Device: ", device)
    return device


def print_gpu_memory_info(summary=False, peak=True):
    
    """
    Print GPU memory summary.
    """

    def get_gpu_peak_memory_usage_gb():
        gb = torch.cuda.max_memory_allocated()/1024**3
        return gb
    
    if summary:
        print("GPU memory summary:")
        print(torch.cuda.memory_summary(abbreviated=True))
    
    if peak:
        print("Peak GPU memory usage:")
        print(f"{get_gpu_peak_memory_usage_gb():.5f} GB")