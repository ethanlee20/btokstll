
import torch


def torch_tensor_from_pandas(dataframe_or_series):

    """
    Convert a pandas dataframe or series to a torch tensor.
    """

    tensor = torch.from_numpy(
        dataframe_or_series.to_numpy()
    )
    return tensor