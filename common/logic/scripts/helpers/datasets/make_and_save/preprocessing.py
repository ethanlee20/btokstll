
from numpy import sqrt

from ..constants import (
    Names_of_Datasets,
    Names_of_Variables,
    Names_of_Labels,
    Names_of_Levels,
    Names_of_q_Squared_Vetos,
    Names_of_Splits,
    Raw_Signal_Trial_Ranges
)
from .aggregated_signal import Aggregated_Signal_Dataframe_Handler
from .aggregated_background import load_raw_charge_and_mix_background_files


def calculate_training_dataset_means_and_stdevs( 
    level, 
    q_squared_veto, 
    path_to_main_datasets_dir, 
    bkg_fraction=None,
    bkg_charge_fraction=None,
    path_to_raw_bkg_dir=None,
):
    
    if (level != Names_of_Levels().detector_and_background) and (bkg_fraction is not None):
        raise ValueError 
    
    aggregated_signal_dataframe_handler = Aggregated_Signal_Dataframe_Handler(
        path_to_main_datasets_dir=path_to_main_datasets_dir,
        level=level,
        trial_range=Raw_Signal_Trial_Ranges().train
    )
    aggregated_signal_dataframe = aggregated_signal_dataframe_handler.get_dataframe()
    aggregated_signal_dataframe = apply_q_squared_veto(
        dataframe=aggregated_signal_dataframe, 
        q_squared_veto=q_squared_veto
    )

    aggregated_signal_means = aggregated_signal_dataframe[Names_of_Variables().list_].mean()
    
    if bkg_fraction is not None:
        
        charge_background_dataframe, mix_background_dataframe = load_raw_charge_and_mix_background_files(
            path_to_raw_bkg_dir=path_to_raw_bkg_dir,
            split=Names_of_Splits().train
        )
        charge_background_dataframe = apply_q_squared_veto(
            dataframe=charge_background_dataframe, 
            q_squared_veto=q_squared_veto
        )
        mix_background_dataframe = apply_q_squared_veto(
            dataframe=mix_background_dataframe,
            q_squared_veto=q_squared_veto
        )

        charge_background_means = charge_background_dataframe[Names_of_Variables().list_].mean()
        mix_background_means = mix_background_dataframe[Names_of_Variables().list_].mean()

        weighted_means = (
            (1 - bkg_fraction) * aggregated_signal_means
            + (bkg_fraction * bkg_charge_fraction) * charge_background_means
            + (bkg_fraction * (1 - bkg_charge_fraction)) * mix_background_means 
        ) 

        weighted_standard_deviations = sqrt(
            (1 - bkg_charge_fraction) * ((aggregated_signal_dataframe[Names_of_Variables().list_] - weighted_means[Names_of_Variables().list_])**2).mean()
            + (bkg_fraction * bkg_charge_fraction) * ((charge_background_dataframe[Names_of_Variables().list_] - weighted_means[Names_of_Variables().list_])**2).mean()
            + (bkg_fraction * (1 - bkg_charge_fraction)) * ((mix_background_dataframe[Names_of_Variables().list_] - weighted_means[Names_of_Variables().list_])**2).mean()
        )

        return weighted_means, weighted_standard_deviations
    
    else:   

        aggregated_signal_stdevs = aggregated_signal_dataframe[Names_of_Variables().list_].std()
        return aggregated_signal_means, aggregated_signal_stdevs


def apply_standard_scale(
    dataframe, 
    list_of_names_of_columns_to_scale,
    level, 
    q_squared_veto, 
    path_to_main_datasets_dir, 
    bkg_fraction=None,
    bkg_charge_fraction=None,
    path_to_raw_bkg_dir=None,
):
    
    print("Applying standand scale.")

    dataframe = dataframe.copy()

    means, stdevs = calculate_training_dataset_means_and_stdevs(
        level=level,
        q_squared_veto=q_squared_veto,
        path_to_main_datasets_dir=path_to_main_datasets_dir,
        bkg_fraction=bkg_fraction,
        bkg_charge_fraction=bkg_charge_fraction,
        path_to_raw_bkg_dir=path_to_raw_bkg_dir
    )

    dataframe[list_of_names_of_columns_to_scale] = (
        (dataframe[list_of_names_of_columns_to_scale] - means[list_of_names_of_columns_to_scale]) 
        / stdevs[list_of_names_of_columns_to_scale]
    )

    print("Applied standard scale.")
    return dataframe


def drop_rows_that_have_a_nan(dataframe, verbose=True):
    
    """
    Drop rows of a dataframe that contain a NaN.
    """

    print("Removing rows that have a NaN.")

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
    'resonances' keeps 0 < q^2 < 20 except for vetos around the resonances.
    """

    def apply_tight_cut(dataframe):
        tight_cut = (
            (dataframe[Names_of_Variables().q_squared] > 1) 
            & (dataframe[Names_of_Variables().q_squared] < 8)
        )
        return dataframe[tight_cut]
    
    def apply_loose_cut(dataframe):
        loose_cut = (
            (dataframe[Names_of_Variables().q_squared] > 0) 
            & (dataframe[Names_of_Variables().q_squared] < 20)
        )
        return dataframe[loose_cut]
    
    def apply_j_psi_resonace_cut(dataframe):
        j_psi_resonance_cut = (
            (dataframe[Names_of_Variables().q_squared] < 9)
            | (dataframe[Names_of_Variables().q_squared] > 10)
        )
        return dataframe[j_psi_resonance_cut]
    
    def apply_psi_2s_resonance_cut(dataframe):
        psi_2s_resonance_cut = (
            (dataframe[Names_of_Variables().q_squared] < 13)
            | (dataframe[Names_of_Variables().q_squared] > 14)
        )
        return dataframe[psi_2s_resonance_cut]
    
    print("Applying q^2 veto.")
        
    if q_squared_veto not in Names_of_q_Squared_Vetos().tuple_:
        raise ValueError

    dataframe = (
        apply_tight_cut(dataframe) if q_squared_veto == Names_of_q_Squared_Vetos().tight
        else apply_loose_cut(dataframe) if q_squared_veto == Names_of_q_Squared_Vetos().loose
        else apply_psi_2s_resonance_cut(apply_j_psi_resonace_cut(apply_loose_cut(dataframe))) 
        if q_squared_veto == Names_of_q_Squared_Vetos().resonances
        else None
    )
    if dataframe is None: raise ValueError

    print("Applied q^2 veto.")
    
    return dataframe.copy()


def reduce_to_label_subset(dataframe, unbinned_label_subset_list_or_id):
    
    """
    Reduce a dataframe to data from specified labels.

    Label subset should be specified with a list of 
    (unbinned) label values or a special subset ID string.
    """

    print("Applying label subset.")

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

    print("Applied label subset.")
    
    return dataframe


def shuffle_rows(
    dataframe, 
    verbose=True
):
    
    dataframe = dataframe.sample(frac=1)
    if verbose:
        print("Shuffled dataframe.")
    return dataframe


def reduce_num_per_label_to_lowest(dataframe, shuffle):

    print("Reducing events per label to lowest per label.")

    num_lowest = dataframe[Names_of_Labels().unbinned].value_counts().values.min()
    get_subset = lambda x : x.iloc[:num_lowest]
    dataframe = (
        dataframe.groupby(Names_of_Labels().unbinned, group_keys=False)[dataframe.columns]
        .apply(get_subset)
    )

    if shuffle:
        dataframe = shuffle_rows(dataframe)

    print("Reduced events per label to lowest per label.")
    
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
        q_squared_veto=settings.common.preprocessing.q_squared_veto
    )

    bkg_fraction = (
        settings.set.bkg_fraction 
        if settings.common.name in Names_of_Datasets().set_based
        else None
    )
    bkg_charge_fraction = (
        settings.set.bkg_charge_fraction 
        if settings.common.name in Names_of_Datasets().set_based
        else None
    )
    dataframe = apply_standard_scale(
        dataframe=dataframe,
        list_of_names_of_columns_to_scale=list_of_variables_to_standard_scale,
        level=settings.common.level,
        q_squared_veto=settings.common.preprocessing.q_squared_veto,
        path_to_main_datasets_dir=settings.common.path_to_main_datasets_dir,
        bkg_fraction=bkg_fraction,
        bkg_charge_fraction=bkg_charge_fraction,
        path_to_raw_bkg_dir=settings.common.path_to_raw_bkg_dir
    )

    if settings.common.preprocessing.shuffle:
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
    
    print("Applying signal preprocessing.")
    
    dataframe = _apply_common_preprocessing(
        dataframe=dataframe, 
        settings=settings, 
        list_of_variables_to_standard_scale=list_of_variables_to_standard_scale,                                   
        verbose=verbose
    )

    if settings.common.preprocessing.label_subset is not None:
        dataframe = reduce_to_label_subset(
            dataframe=dataframe,
            unbinned_label_subset_list_or_id=settings.common.preprocessing.label_subset
        )

    if settings.common.preprocessing.uniform_label_counts:
        dataframe = reduce_num_per_label_to_lowest(dataframe, shuffle=settings.common.preprocessing.shuffle)
        
    print("Applied signal preprocessing.")

    return dataframe


def apply_background_preprocessing(
    dataframe,
    settings,
    list_of_variables_to_standard_scale,
    verbose=True
):
    
    print("Applying background preprocessing.")
    
    dataframe = _apply_common_preprocessing(
        dataframe=dataframe, 
        settings=settings, 
        list_of_variables_to_standard_scale=list_of_variables_to_standard_scale,                                   
        verbose=verbose
    )

    print("Applied background preprocessing.")
    
    return dataframe














