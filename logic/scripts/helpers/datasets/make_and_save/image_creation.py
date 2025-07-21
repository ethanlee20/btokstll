
from numpy import pi
from scipy.stats import binned_statistic_dd
import torch


def make_image(feature_set_tensor, num_bins_per_dimension):
    
    """
    Make an image like Shawn.

    Variables in the input tensor must be (in this order):             
    q^2, cosine theta mu, cosine theta K, chi

    Return an image (torch tensor). Image is binned in the
    angular variables and contains the average q^2 value per bin. 
    Empty bins are set to 0.
    
    Image has shape of (1, num bins, num bins, num bins).
    """

    feature_set_tensor = (
        feature_set_tensor
        .detach()
        .numpy()
    )

    angular_features = feature_set_tensor[:, 1:]
    q_squared_features = feature_set_tensor[:, 0]

    numpy_image = binned_statistic_dd(
        sample=angular_features,
        values=q_squared_features, 
        bins=num_bins_per_dimension,
        range=[(-1, 1), (-1, 1), (0, 2*pi)]
    ).statistic
    
    torch_image = torch.from_numpy(numpy_image)
    torch_image = torch.nan_to_num(torch_image, nan=0.0)
    torch_image = torch_image.unsqueeze(dim=0)
    return torch_image
