
"""
Dataset utilities
"""


import pathlib
import numpy
import torch
import pandas
import scipy


def drop_na(df, verbose=True):

    """
    Drop rows of a dataframe that contain a NaN.

    Parameters
    ----------
    df : pandas.DataFrame
        The original dataframe.

    Returns
    -------
    df_out : pandas.DataFrame
        A modified copy of the dataframe.
    """

    if verbose:
        print("Number of NA values: ", df.isna().sum())
    df_out = df.dropna()
    if verbose:
        print("Removed NA rows.")
    return df_out


def apply_q_squared_veto(df: pandas.DataFrame, strength:str):
    
    """
    Apply a q^2 veto to a dataframe of B->K*ll events.

    Parameters
    ---------- 
    df : pandas.DataFrame
        The dataframe of events.
    strength : str
        The strength of the veto. ('tight' or 'loose') 
        Tight keeps  1 < q^2 < 8.
        Loose keeps 0 < q^2 < 20.

    Returns
    -------
    df_vetoed : pandas.DataFrame
        The reduced dataframe.
        Returns a copy.
    """

    if strength not in ("tight", "loose"):
        raise ValueError(
            "strength must be 'tight' or 'loose'"
        )

    tight_bounds = (1, 8) 
    loose_bounds = (0, 20)

    bounds = (
        tight_bounds if strength == "tight"
        else loose_bounds if strength == "loose"
        else None
    )

    lower_bound = bounds[0]
    upper_bound = bounds[1]

    df_vetoed = df[
        (df["q_squared"]>lower_bound) 
        & (df["q_squared"]<upper_bound)
    ].copy()
    return df_vetoed


def balance_classes(
    df: pandas.DataFrame, 
    label_column_name: str
):
    
    """
    Reduce the number of events per unique label 
    to the minimum over the labels.

    Shuffles dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        The original dataframe.
    label_column_name : str
        The name of the column containing labels.

    Returns
    -------
    df_balanced : pandas.DataFrame
        The balanced dataframe. 
        All data is copied.

    """
    df_shuffled = df.sample(frac=1)
    group_by_label = df_shuffled.groupby(
        label_column_name
    )
    num_events = [
        len(df_label) 
        for _, df_label in group_by_label
    ]
    min_num_events = min(num_events)
    list_df_balanced = [
        df_label[:min_num_events] 
        for _, df_label in group_by_label
    ]
    df_balanced = pandas.concat(list_df_balanced)
    return df_balanced


def get_dataset_prescale(
    kind:str, 
    level:str, 
    q_squared_veto:str, 
    var:str,
):
    """
    Get prescale values calculated from the
    first 20 trials (training dataset).

    Prescale values are used for standard scaling
    the dataset's features.

    Parameters
    ----------
    kind : str
        The kind of value to get.
        "mean" or "std".
    level : str
        The reconstruction level.
        "gen" or "det".
        Determines which dataset's values to get.
    q_squared_veto : str
        "tight" or "loose". "tight" gets values computed
        from the dataset with a tight q^2 veto.
        "loose" gets values computed from the dataset
        with a loose q^2 veto.
    var : str
        The variable whose value to get.
        "q_squared", "costheta_mu", "costheta_K" 
        or "chi"

    Returns 
    -------
    value : float
        The requested prescale value.

    """

    table_tight_q2_veto = {
        "mean": {
            "gen": {
                "q_squared": 4.752486,
                "costheta_mu": 0.065137,
                "costheta_K": 0.000146,
                "chi": 3.141082, 
            }, 
            "det": {
                "q_squared": 4.943439,
                "costheta_mu": 0.077301,
                "costheta_K": -0.068484,
                "chi": 3.141730, 
            }
        },
        "std": {
            "gen": {
                "q_squared": 2.053569,
                "costheta_mu": 0.505880,
                "costheta_K": 0.694362,
                "chi": 1.811370, 
            }, 
            "det": {
                "q_squared": 2.030002,
                "costheta_mu": 0.463005,
                "costheta_K": 0.696061,
                "chi": 1.830277, 
            }
        }
    }

    table_loose_q2_veto = {
        "mean": {
            "gen": {
                "q_squared": 9.248781,
                "costheta_mu": 0.151290,
                "costheta_K": 0.000234,
                "chi": 3.141442, 
            }, 
            "det": {
                "q_squared": 10.353162,
                "costheta_mu": 0.177404,
                "costheta_K": -0.031136,
                "chi": 3.141597, 
            }
        },
        "std": {
            "gen": {
                "q_squared": 5.311177,
                "costheta_mu": 0.524446,
                "costheta_K": 0.635314,
                "chi": 1.803100, 
            }, 
            "det": {
                "q_squared": 5.242896,
                "costheta_mu": 0.508787,
                "costheta_K": 0.622743,
                "chi": 1.820018, 
            }    
        }
    }

    table = (
        table_tight_q2_veto if q_squared_veto == "tight"
        else table_loose_q2_veto if q_squared_veto == "loose"
        else None
    )

    value = table[kind][level][var]
    return value


def bootstrap_labeled_sets(
    x:torch.Tensor, 
    y:torch.Tensor, 
    n:int, 
    m:int, 
    reduce_labels:bool=True, 
    labels_to_sample:list[float]=None
):
    
    """
    Bootstrap m sets of n examples for each unique label.

    Parameters
    ----------
    x : torch.Tensor
        Array of features of shape: 
        (number of events, number of features)
    y : torch.Tensor
        Array of labels of shape: (number of events)
    n : int
        Number of events per bootstrap  
    m : int
        Number of bootstraps per unique label  
    reduce_labels : bool
        Whether or not to return one label per bootstrap
        (versus one label per event).
    labels_to_sample : list
        Subset of label values to bootstrap.
        If None, bootstrap from all unique label values.

    Returns
    -------
    bootstrap_x : torch.Tensor
        Torch tensor of bootstrapped features.
        Shape (m, n, number of features).
    bootstrap_y : torch.Tensor
        Torch tenor of bootstrapped labels.
        Shape (m) if reduce_labels is True.
        Shape (m, n) if reduce_labels is False.
    """

    bootstrap_x = []
    bootstrap_y = []
    
    if labels_to_sample is None:
        labels_to_sample = torch.unique(
            y, 
            sorted=True
        ) 
    else:
        labels_to_sample, labels_to_sample_counts = (
            torch.unique(
                labels_to_sample, 
                sorted=True,
                return_counts=True,
            )
        )
        if torch.any(labels_to_sample_counts > 1):
            raise ValueError(
                "Cannot bootstrap from subset " \
                "containing repeated labels."
            )

    for label in labels_to_sample:    

        for _ in range(m):

            pool_x = x[y==label]
            pool_y = y[y==label]
            assert pool_x.shape[0] == pool_y.shape[0]

            selection_indices = torch.randint(
                low=0, 
                high=len(pool_x), 
                size=(n,)
            )

            _bootstrap_x = pool_x[selection_indices]
            _bootstrap_y = pool_y[selection_indices]

            bootstrap_x.append(_bootstrap_x.unsqueeze(0))
            bootstrap_y.append(_bootstrap_y.unsqueeze(0))

    bootstrap_x = torch.concatenate(bootstrap_x)
    bootstrap_y = torch.concatenate(bootstrap_y)

    if reduce_labels:
        bootstrap_y = (
            torch.unique_consecutive(
                bootstrap_y, dim=1
            ).squeeze()
        )
        assert (
            bootstrap_y.shape[0] 
            == bootstrap_x.shape[0]
        )

    return bootstrap_x, bootstrap_y




def make_image(ar_feat, n_bins):

    """
    Make an image of a B->K*ll dataset. 
    Like Shawn.

    Order of input features must be:             
        "q_squared", 
        "costheta_mu", 
        "costheta_K", 
        "chi"

    Parameters
    ----------
    ar_feat : array_like
        Array of features.
    n_bins : int
        Number of bins per dimension.

    Returns
    -------
    image : torch.Tensor
        Torch tensor of the average q^2 calculated 
        per 3D angular bin.
        Shape of (1, n_bins, n_bins, n_bins).
        A 3D image with 1 color channel.
        Empty bins set to 0.
    """

    angular_features = ar_feat[:,1:]
    q_squared_features = ar_feat[:,0]
    np_image = scipy.stats.binned_statistic_dd(
        sample=angular_features,
        values=q_squared_features, 
        bins=n_bins,
        range=[(-1, 1), (-1, 1), (0, 2*numpy.pi)]
    ).statistic
    torch_image = torch.from_numpy(np_image)
    torch_image = torch.nan_to_num(torch_image)
    torch_image = torch_image.unsqueeze(0)
    return torch_image


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



    
