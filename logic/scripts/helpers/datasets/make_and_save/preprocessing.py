
from ..constants import (
    Names_of_Variables,
    Names_of_Labels,
    Names_of_Levels,
    Names_of_q_Squared_Vetos
)


def get_dataset_mean_value(variable, level, q_squared_veto):
    
    mean_values = {
        Names_of_q_Squared_Vetos().tight: {
            Names_of_Levels().generator: {
                Names_of_Variables().q_squared: 4.752486,
                Names_of_Variables().cos_theta_mu: 0.065137,
                Names_of_Variables().cos_k: 0.000146,
                Names_of_Variables().chi: 3.141082, 
            }, 
            Names_of_Levels().detector: {
                Names_of_Variables().q_squared: 4.943439,
                Names_of_Variables().cos_theta_mu: 0.077301,
                Names_of_Variables().cos_k: -0.068484,
                Names_of_Variables().chi: 3.141730, 
            },
            Names_of_Levels().detector_and_background: {
                Names_of_Variables().q_squared: 4.944093,
                Names_of_Variables().cos_theta_mu: 0.078201,
                Names_of_Variables().cos_k: -0.068596,
                Names_of_Variables().chi: 3.141794, 
            }
        },
        Names_of_q_Squared_Vetos().loose: {
            Names_of_Levels().generator: {
                Names_of_Variables().q_squared: 9.248781,
                Names_of_Variables().cos_theta_mu: 0.151290,
                Names_of_Variables().cos_k: 0.000234,
                Names_of_Variables().chi : 3.141442, 
            }, 
            Names_of_Levels().detector: {
                Names_of_Variables().q_squared: 10.353162,
                Names_of_Variables().cos_theta_mu: 0.177404,
                Names_of_Variables().cos_k: -0.031136,
                Names_of_Variables().chi: 3.141597, 
            },
            Names_of_Levels().detector_and_background: {  
                Names_of_Variables().q_squared: 10.134447,
                Names_of_Variables().cos_theta_mu: 0.182426,
                Names_of_Variables().cos_k: -0.044501,
                Names_of_Variables().chi: 3.141522, 
            }
        }
    }

    selected_mean_value = mean_values[q_squared_veto][level][variable]
    return selected_mean_value


def get_dataset_standard_deviation(variable, level, q_squared_veto):
    
    stdev_values = {
        Names_of_q_Squared_Vetos().tight: {
            Names_of_Levels().generator: {
                Names_of_Variables().q_squared: 2.053569,
                Names_of_Variables().cos_theta_mu: 0.505880,
                Names_of_Variables().cos_k: 0.694362,
                Names_of_Variables().chi: 1.811370, 
            }, 
            Names_of_Levels().detector: {
                Names_of_Variables().q_squared: 2.030002,
                Names_of_Variables().cos_theta_mu: 0.463005,
                Names_of_Variables().cos_k: 0.696061,
                Names_of_Variables().chi: 1.830277, 
            },
            Names_of_Levels().detector_and_background: {
                Names_of_Variables().q_squared: 2.029607,
                Names_of_Variables().cos_theta_mu: 0.463519,
                Names_of_Variables().cos_k: 0.695645,
                Names_of_Variables().chi: 1.830584, 
            }
        },
        Names_of_q_Squared_Vetos().loose: {
            Names_of_Levels().generator: {
                Names_of_Variables().q_squared: 5.311177,
                Names_of_Variables().cos_theta_mu: 0.524446,
                Names_of_Variables().cos_k: 0.635314,
                Names_of_Variables().chi : 1.803100, 
            }, 
            Names_of_Levels().detector: {
                Names_of_Variables().q_squared: 5.242896,
                Names_of_Variables().cos_theta_mu: 0.508787,
                Names_of_Variables().cos_k: 0.622743,
                Names_of_Variables().chi: 1.820018, 
            },
            Names_of_Levels().detector_and_background: {  
                Names_of_Variables().q_squared: 4.976700,
                Names_of_Variables().cos_theta_mu: 0.523063,
                Names_of_Variables().cos_k: 0.615523,
                Names_of_Variables().chi: 1.831986, 
            }
        }
    }

    selected_stdev_value = stdev_values[q_squared_veto][level][variable]
    return selected_stdev_value


def apply_standard_scale(
    dataframe, 
    list_of_variables,
    level, 
    q_squared_veto
):
    
    """
    Standard scale columns of a dataframe.

    Outputs are given as:
    (original value - mean) / standard deviation
    """

    dataframe = dataframe.copy()

    for variable in list_of_variables:
        mean_value = get_dataset_mean_value(
            variable=variable, 
            level=level, 
            q_squared_veto=q_squared_veto
        )
        standard_deviation = get_dataset_standard_deviation(
            variable=variable,
            level=level,
            q_squared_veto=q_squared_veto
        )
        dataframe[variable] = (
            (dataframe[variable] - mean_value) 
            / standard_deviation
        )

    return dataframe


def drop_rows_that_have_a_nan(dataframe, verbose=True):
    
    """
    Drop rows of a dataframe that contain a NaN.
    """

    dataframe = dataframe.copy()

    if verbose:
        print(
            "Number of NA values: \n", 
            dataframe.isna().sum()
        )
    dataframe = dataframe.dropna()
    if verbose:
        print("Removed rows that have a NaN.")

    return dataframe


def apply_q_squared_veto(dataframe, q_squared_veto):
    
    """
    Apply a q^2 veto to a dataframe of B->K*ll events.
    'tight' keeps  1 < q^2 < 8.
    'loose' keeps 0 < q^2 < 20.
    """

    if q_squared_veto not in Names_of_q_Squared_Vetos().tuple_:
        raise ValueError
    
    tight_bounds = (1, 8) 
    loose_bounds = (0, 20)
    
    bounds = (
        tight_bounds if q_squared_veto == Names_of_q_Squared_Vetos().tight
        else loose_bounds if q_squared_veto == Names_of_q_Squared_Vetos().loose
        else None
    )
    if bounds is None:
        raise ValueError
    lower_bound = bounds[0]
    upper_bound = bounds[1]
    
    dataframe = dataframe[
        (dataframe[Names_of_Variables().q_squared] > lower_bound) 
        & (dataframe[Names_of_Variables().q_squared] < upper_bound)
    ].copy()
    return dataframe


def reduce_to_label_subset(dataframe, unbinned_label_subset_list_or_id):
    
    """
    Reduce a dataframe to data from specified labels.

    Label subset should be specified with a list of 
    (unbinned) label values or a special subset ID string.
    """

    less_than_or_equal_to_zero_subset_id = "less than or equal to zero"
    
    dataframe = dataframe.copy()

    labels_column = dataframe[Names_of_Labels().unbinned]

    if type(unbinned_label_subset_list_or_id) == str:
        subset_id = unbinned_label_subset_list_or_id
        if subset_id == less_than_or_equal_to_zero_subset_id:
            dataframe = dataframe[labels_column <= 0]
        else: raise ValueError

    elif type(unbinned_label_subset_list_or_id) == list:
        subset_list = unbinned_label_subset_list_or_id
        dataframe = dataframe[labels_column.isin(subset_list)]
    
    else: raise ValueError
    
    return dataframe


def shuffle_rows(
    dataframe, 
    verbose=True
):

    dataframe = dataframe.sample(frac=1)
    if verbose:
        print("Shuffled dataframe.")
    return dataframe


def _apply_common_preprocessing(
    dataframe,
    settings,
    list_of_variables_to_standard_scale,
    verbose=True
):

    dataframe = drop_rows_that_have_a_nan(
        dataframe=dataframe,
        verbose=verbose
    )

    dataframe = apply_q_squared_veto(
        dataframe=dataframe, 
        q_squared_veto=settings.preprocessing.q_squared_veto
    )

    dataframe = apply_standard_scale(
        dataframe=dataframe,
        list_of_variables=list_of_variables_to_standard_scale,
        level=settings.common.level,
        q_squared_veto=settings.preprocessing.q_squared_veto
    )

    if settings.preprocessing.shuffle:
        dataframe = shuffle_rows(
            dataframe=dataframe,
            verbose=verbose
        )
    
    return dataframe


def apply_signal_preprocessing(
    dataframe,
    settings,
    list_of_variables_to_standard_scale,
    verbose=True
):
    
    dataframe = _apply_common_preprocessing(
        dataframe=dataframe, 
        settings=settings, 
        list_of_variables_to_standard_scale=list_of_variables_to_standard_scale,                                   
        verbose=verbose
    )
    
    if settings.preprocessing.label_subset is not None:
        dataframe = reduce_to_label_subset(
            dataframe=dataframe,
            unbinned_label_subset_list_or_id=settings.preprocessing.label_subset
        )
        
    return dataframe


def apply_background_preprocessing(
    dataframe,
    settings,
    list_of_variables_to_standard_scale,
    verbose=True
):
    
    dataframe = _apply_common_preprocessing(
        dataframe=dataframe, 
        settings=settings, 
        list_of_variables_to_standard_scale=list_of_variables_to_standard_scale,                                   
        verbose=verbose
    )
    return dataframe














