
"""
Dataset utilities.
"""


import torch


def get_num_per_unique_label(labels:torch.Tensor):

    """
    Get the number of examples with each unique label.

    Only works if there is the same number
    of examples per each unique label.

    Parameters
    ----------
    labels : torch.Tensor
        Array of labels. One label per example.
    
    Returns
    -------
    num_per_label : int
        Number of examples of each unique label.
    """

    _, label_counts = torch.unique(
        labels, 
        return_counts=True
    )
    # check same number of sets per label
    assert torch.all(label_counts == label_counts[0]) 
    num_per_label = label_counts[0].item()
    return num_per_label



    
