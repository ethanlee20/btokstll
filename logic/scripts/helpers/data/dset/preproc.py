
"""
Dataset preprocessing utilities.
"""

import numpy
import torch
import pandas
import scipy

from .config import Config_Dataset
from .constants import (
    Names_Datasets,
    Names_Variables,
    Names_Labels,
    Names_Levels,
    Names_q_Squared_Vetos
)


def get_dataset_prescale(
    kind:str,  
    level:str, 
    q_squared_veto:str, 
    var:str,
):
    
    """
    Get prescale value calculated from the
    first 20 trials (training dataset).

    Prescale values are used for standard scaling
    the dataset's features.
    """

    if kind not in ("mean", "std"):
        raise ValueError(
            "kind not known. "
            f"Must be in {("mean", "std")}."
        )
    if level not in Names_Levels().tuple_:
        raise ValueError(
            "Level not known. "
            f"Must be in {Names_Levels().tuple_}."
        )
    if q_squared_veto not in Names_q_Squared_Vetos().tuple_:
        raise ValueError(
            "q_squared_veto not known. "
            f"Must be in {Names_q_Squared_Vetos().tuple_}."
        )
    if var not in Names_Variables().tuple_:
        raise ValueError(
            "var not known. "
            f"Must be in {Names_Variables().tuple_}."
        )

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
            },
            "det_bkg": {
                "q_squared": 4.944093,
                "costheta_mu": 0.078201,
                "costheta_K": -0.068596,
                "chi": 3.141794, 
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
            },
            "det_bkg": {
                "q_squared": 2.029607,
                "costheta_mu": 0.463519,
                "costheta_K": 0.695645,
                "chi": 1.830584, 
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
            },
            "det_bkg": {
                "q_squared": 10.134447,
                "costheta_mu": 0.182426,
                "costheta_K": -0.044501,
                "chi": 3.141522, 
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
            },
            "det_bkg": {
                "q_squared": 4.976700,
                "costheta_mu": 0.523063,
                "costheta_K": 0.615523,
                "chi": 1.831986, 
            }
        }
    }

    table = (
        table_tight_q2_veto 
        if q_squared_veto == "tight"
        else table_loose_q2_veto 
        if q_squared_veto == "loose"
        else None
    )

    value = table[kind][level][var]
    return value


def apply_standard_scale(
    df, 
    level, 
    q_squared_veto, 
    columns
):
    
    """
    Standard scale columns of a dataframe.
    Mean and standard deviation are precalculated.

    Outputs are given as:
    (original_value - mean) / standard_deviation
    """

    for column in columns:
        if column not in (
            "q_squared", 
            "costheta_mu", 
            "costheta_K", 
            "chi"
        ):
            raise ValueError(
                "Column name not recognized: "
                f"{column}"
            )
        
    for column in columns:
        df[column] = ( 
            (
                df[column] 
                - get_dataset_prescale(
                    "mean", 
                    level, 
                    q_squared_veto, 
                    column,
                )
            ) 
            / get_dataset_prescale(
                "std", 
                level, 
                q_squared_veto, 
                column,
            )
        )
    return df


def apply_balance_classes(
    df: pandas.DataFrame, 
    name_label: str
):
    """
    Reduce the number of events per unique label 
    to the minimum over the labels.

    Shuffles dataframe.
    """

    df_shuffled = df.sample(frac=1)
    
    group_by_label = df_shuffled.groupby(
        name_label
    )

    list_num_events = [
        len(df_label) 
        for _, df_label in group_by_label
    ]

    min_num_events = min(list_num_events)
    
    list_df_balanced = [
        df_label[:min_num_events] 
        for _, df_label in group_by_label
    ]

    df_balanced = pandas.concat(list_df_balanced)
    
    return df_balanced


def apply_drop_na(
    df, 
    verbose=True
):
    
    """
    Drop rows of a dataframe that contain a NaN.
    """

    if verbose:
        print(
            "Number of NA values: \n", 
            df.isna().sum()
        )

    df_out = df.dropna()
    
    if verbose:
        print("Removed NA rows.")
    
    return df_out


def apply_q_squared_veto(
    df: pandas.DataFrame, 
    strength:str
):
    
    """
    Apply a q^2 veto to a dataframe of B->K*ll events.
    'tight' keeps  1 < q^2 < 8.
    'loose' keeps 0 < q^2 < 20.
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
        (df[Names_Variables().q_squared] > lower_bound) 
        & (df[Names_Variables().q_squared] < upper_bound)
    ].copy()
    return df_vetoed


def apply_label_subset(
    df, 
    name_label:str, 
    label_subset:list,
):
    
    """
    Reduce a dataframe to data from specified labels.
    """

    df = df[
        df[name_label]
        .isin(label_subset)
    ]
    return df


def apply_shuffle(
    df, 
    verbose=True
):
    
    """
    Shuffle rows of a dataframe.
    """

    df = df.sample(frac=1)

    if verbose:
        print("Shuffled dataframe.")

    return df


def apply_cleaning_signal(
    df, 
    config:Config_Dataset, 
    bin_map=None, 
    verbose=True,
):
    
    """
    Apply cleaning to aggregated signal dataframe.
    Includes q^2 veto, standard scaling, 
    balancing classes, label subset, 
    dropping NA rows, and shuffling.
    """

    def check_input():
        if config.is_binned and (bin_map is None):
            raise ValueError(
                "bin_map must be specified for binned dataset."
            )
    check_input()

    df = df.copy()
    
    df = apply_q_squared_veto(
        df, 
        config.q_squared_veto,
    )    

    if config.std_scale:
        features_to_scale = (
            Names_Variables().tuple_ if (
                config.name != Names_Datasets().images
            )
            else [Names_Variables().q_squared] if (
                config.name == Names_Datasets().images
            )
            else None
        )
        df = apply_standard_scale(
            df, 
            config.level, 
            config.q_squared_veto,
            features_to_scale,
        )

    if config.balanced_classes:
        df = apply_balance_classes(
            df, 
            config.name_label,
        )

    if config.label_subset:
        label_subset = (
            [
                (bin_map == label).nonzero().item() 
                for label in config.label_subset
            ]
            if config.is_binned
            else config.label_subset
        )
        df = apply_label_subset(
            df,
            config.name_label,
            label_subset,
        )

    df = apply_drop_na(df)

    if config.shuffle:
        df = apply_shuffle(df)

    if verbose:
        print("Applied cleaning.")
    
    return df


def apply_cleaning_bkg(
    df,
    config,
):
    
    """
    Apply cleaning to background dataset.
    """

    df = df.copy()

    df = apply_q_squared_veto(
        df, 
        config.q_squared_veto,
    )

    if config.std_scale:

        features_to_scale = (
            Names_Variables().list_ if (
                config.name != Names_Datasets().images
            )
            else [Names_Variables().q_squared] if (
                config.name == Names_Datasets().images
            )
            else None
        )

        df = apply_standard_scale(
            df, 
            config.level, 
            config.q_squared_veto,
            features_to_scale,
        )
    
    df = apply_drop_na(df)
    
    if config.shuffle:
        df = apply_shuffle(df)

    print("Applied cleaning.")

    return df


def convert_to_binned(
    df, 
    name_label_unbinned, 
    name_label_binned
):
    
    """
    Convert dataframe unbinned labels 
    to binned labels.
    """

    def to_bins(
        ar
    ):
        """
        Translate values in an array to bin numbers.

        Each unique value in the array corresponds to
        a unique bin number.

        Parameters
        ----------
        ar : list | numpy.ndarray | pandas.Series

        Returns
        -------
        bins : numpy.ndarray
            Array of bin numbers.
            A bin number is assigned for each
            value in the original array.
        bin_map : numpy.ndarray
            Map between original values and
            bin numbers.
            Indices of this array correspond
            to bin numbers.
            Values of this array are original values.
        """

        ar = numpy.array(ar)

        bin_map, inverse_indices = numpy.unique(
            ar, 
            return_inverse=True
        )

        bin_indices = numpy.arange(len(bin_map))

        bins = bin_indices[inverse_indices]

        return bins, bin_map
    
    bins, bin_map = to_bins(
        df[name_label_unbinned]
    )

    df[name_label_binned] = bins

    df = df.drop(
        columns=name_label_unbinned
    )

    bin_map = torch.from_numpy(bin_map)

    return df, bin_map


def bins_to_probs(
    bins,
    num_bins,
):
    
    """
    convert bin index to one-hot probabilities over bins.
    """
    
    num_examples = len(bins)

    probs = torch.zeros(
        (num_examples, num_bins),
        dtype=torch.float32,
    )
    probs[
        torch.arange(num_examples), 
        bins.int()
    ] = 1.0

    return probs


def values_from_probs(probs, bin_map):
    
    values = bin_map[(probs==1.0).nonzero()[:, 1]]
    return values


def bkg_probs(num_events, num_bins):

    """
    Make background probability labels.
    """

    probs = (
        torch.ones((num_events, num_bins))
        / num_bins
    )
    return probs 


def bootstrap_labeled_sets(
    features:torch.Tensor, 
    labels:torch.Tensor, 
    num_events_per_set:int, 
    num_sets_per_label:int, 
    reduce_labels:bool=True, 
):  
    
    """
    Bootstrap m sets of n examples for each unique label.

    Parameters
    ----------
    features : torch.Tensor
        Array of features of shape: 
        (number of events, number of features)
    labels : torch.Tensor
        Array of labels of shape: (number of events)
    num_events_per_set : int
        Number of events per bootstrap  
    num_sets_per_label : int
        Number of bootstraps per unique label  
    reduce_labels : bool
        Whether or not to return one label per bootstrap
        (versus one label per event).

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
    
    labels_to_sample = torch.unique(
        labels, 
        sorted=True
    )

    for label in labels_to_sample:    

        for _ in range(num_sets_per_label):

            pool_x = features[labels==label]
            pool_y = labels[labels==label]
            assert pool_x.shape[0] == pool_y.shape[0]

            selection_indices = torch.randint(
                low=0, 
                high=len(pool_x), 
                size=(num_events_per_set,)
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


def bootstrap_bkg(
    df_charge,
    df_mix,
    num_events_per_set,
    num_sets,
    frac_charge,
):
    
    """
    Bootstrap sets of background events.
    """
    
    num_charge = int(
        num_events_per_set 
        * frac_charge
    )
    num_mix = num_events_per_set - num_charge

    list_tensors_bootstrap = []
    for _ in range(num_sets):
        tensor_mix = torch_from_pandas(
            df_mix.sample(n=num_mix, replace=True)
        )
        tensor_charge = torch_from_pandas(
            df_charge.sample(n=num_charge, replace=True)
        )
        _tensor_combo = torch.unsqueeze(
            torch.concat(
                [
                    tensor_mix,
                    tensor_charge,
                ]
            ),
            dim=0,
        )
        list_tensors_bootstrap.append(_tensor_combo)

    tensor_bootstrap = torch.concat(
        list_tensors_bootstrap
    )
    return tensor_bootstrap


def make_image(
    ar_feat,
    n_bins,
):
    """
    Make an image of a B->K*ll dataset. 
    Like Shawn.

    Order of input features in array must be:             
        q^2, 
        cos theta mu, 
        cos theta K, 
        chi

    Parameters
    ----------
    ar_feat : torch.Tensor
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

    ar_feat = ar_feat.detach().numpy()

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


def torch_from_pandas(
    obj,
):
    """
    Convert a pandas object to a torch tensor.

    (Object can be a dataframe or series).
    """

    obj = torch.from_numpy(
        obj.to_numpy()
    )
    return obj
