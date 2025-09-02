
from collections import namedtuple

from numpy import linspace, sqrt, nan, pi
from pandas import cut

from ..datasets.constants import Names_of_Variables, Names_of_Labels


def bin_data(
    dataframe,
    name_of_binning_variable, 
    start, 
    stop, 
    num_bins,
    return_bin_edges=False,
):
    
    bin_edges = linspace(start=start, stop=stop, num=num_bins+1)
    bins = cut(dataframe[name_of_binning_variable], bin_edges, include_lowest=True) # the interval each event falls into
    groupby_binned = dataframe.groupby(bins, observed=False)

    if return_bin_edges:
        return groupby_binned, bin_edges
    return groupby_binned


def calc_bin_middles(start, stop, num_bins):

    bin_edges, step = linspace(
        start=start,
        stop=stop,
        num=num_bins+1,
        retstep=True,
    )
    bin_middles = bin_edges[:-1] + step/2 
    return bin_middles


def calc_afb_of_q_squared(dataframe, num_bins):

    """
    Calcuate Afb as a function of q squared.
    Afb is the forward-backward asymmetry.
    """

    def get_column_cos_theta_mu(df):
        column = df[Names_of_Variables().cos_theta_mu]
        return column
    
    def calc_num_forward(df):
        series_cos_theta_mu = get_column_cos_theta_mu(df)
        f = series_cos_theta_mu[(series_cos_theta_mu > 0) & (series_cos_theta_mu < 1)].count()
        return f
    
    def calc_num_backward(df):
        series_cos_theta_mu = get_column_cos_theta_mu(df)
        b = series_cos_theta_mu[(series_cos_theta_mu > -1) & (series_cos_theta_mu < 0)].count()
        return b

    def _calc_afb(df):
        f = calc_num_forward(df)
        b = calc_num_backward(df)
        afb = (f - b) / (f + b)
        return afb

    def _calc_afb_err(df):
        f = calc_num_forward(df)
        b = calc_num_backward(df)
        f_stdev = sqrt(f)
        b_stdev = sqrt(b)
        afb_stdev = 2*f*b / (f+b)**2 * sqrt((f_stdev/f)**2 + (b_stdev/b)**2)
        afb_err = afb_stdev
        return afb_err

    q_squared_lower_bound = 0
    q_squared_upper_bound = 20

    dataframe = dataframe[
        (dataframe[Names_of_Variables().q_squared]>q_squared_lower_bound) 
        & (dataframe[Names_of_Variables().q_squared]<q_squared_upper_bound)
    ]

    groupby_binned = bin_data(
        dataframe=dataframe, 
        name_of_binning_variable=Names_of_Variables().q_squared, 
        start=q_squared_lower_bound,
        stop=q_squared_upper_bound,
        num_bins=num_bins,     
    )
    
    afbs = groupby_binned.apply(_calc_afb)
    errs = groupby_binned.apply(_calc_afb_err)

    bin_middles = calc_bin_middles(
        start=q_squared_lower_bound, 
        stop=q_squared_upper_bound, 
        num_bins=num_bins,
    )

    return bin_middles, afbs, errs


def calc_afb_of_q_squared_over_delta_C9_values(dataframe, num_bins):

    delta_C9_values = []
    bin_middles_over_delta_C9_values = []
    afbs_over_delta_C9_values = []
    afb_errors_over_delta_C9_values = []

    for delta_C9_value, subset_dataframe in (dataframe.groupby(Names_of_Labels().unbinned)):

        delta_C9_values.append(delta_C9_value)
        bin_middles, afbs, afb_errs = calc_afb_of_q_squared(
            dataframe=subset_dataframe, 
            num_bins=num_bins,
        )
        bin_middles_over_delta_C9_values.append(bin_middles)
        afbs_over_delta_C9_values.append(afbs)
        afb_errors_over_delta_C9_values.append(afb_errs)

    AFB_Results_Over_Delta_C9_Values = namedtuple(
        'AFB_Results_Over_Delta_C9_Values',
        [
            'delta_C9_values',
            'bin_middles_over_delta_C9_values',
            'afbs_over_delta_C9_values',
            'afb_errors_over_delta_C9_values'
        ] 
    )

    results = AFB_Results_Over_Delta_C9_Values(
        delta_C9_values=delta_C9_values,
        bin_middles_over_delta_C9_values=bin_middles_over_delta_C9_values,
        afbs_over_delta_C9_values=afbs_over_delta_C9_values,
        afb_errors_over_delta_C9_values=afb_errors_over_delta_C9_values
    )
    return results


def calc_s5_of_q_squared(dataframe, num_bins):

    def get_column(df, name_variable):
        column = df[name_variable]
        return column

    def get_column_cos_theta_k(df):
        column = get_column(df, Names_of_Variables().cos_k)
        return column
    
    def get_column_chi(df):
        column = get_column(df, Names_of_Variables().chi)
        return column
    
    def calc_num_forward(df):
        series_cos_theta_k = get_column_cos_theta_k(df)
        series_chi = get_column_chi(df)
        num = df[
            (((series_cos_theta_k > 0) & (series_cos_theta_k < 1)) & ((series_chi > 0) & (series_chi < pi/2)))
            | (((series_cos_theta_k > 0) & (series_cos_theta_k < 1)) & ((series_chi > 3*pi/2) & (series_chi < 2*pi)))
            | (((series_cos_theta_k > -1) & (series_cos_theta_k < 0)) & ((series_chi > pi/2) & (series_chi < 3*pi/2)))
        ].count().min()
        return num

    def calc_num_backward(df):
        series_cos_theta_k = get_column_cos_theta_k(df)
        series_chi = get_column_chi(df)
        num = df[
            (((series_cos_theta_k > 0) & (series_cos_theta_k < 1)) & ((series_chi > pi/2) & (series_chi < 3*pi/2)))
            | (((series_cos_theta_k > -1) & (series_cos_theta_k < 0)) & ((series_chi > 0) & (series_chi < pi/2)))
            | (((series_cos_theta_k > -1) & (series_cos_theta_k < 0)) & ((series_chi > 3*pi/2) & (series_chi < 2*pi)))
        ].count().min()
        return num

    def calc_s5(df):
        f = calc_num_forward(df)
        b = calc_num_backward(df)
        try: 
            s5 = 4/3 * (f - b) / (f + b)
        except ZeroDivisionError:
            print("s5 calc: division by 0, returning nan")
            s5 = nan
        return s5

    def calc_s5_err(df):
        """
        Calculate the error of S_5.

        The error is calculated by assuming the "forward" and "backward"
        regions have uncorrelated Poisson errors and propagating
        the errors.
        """
        f = calc_num_forward(df)
        b = calc_num_backward(df)
        f_stdev = sqrt(f)
        b_stdev = sqrt(b)
        try: 
            stdev = 4/3 * 2*f*b / (f+b)**2 * sqrt((f_stdev/f)**2 + (b_stdev/b)**2)
            err = stdev
        except ZeroDivisionError:
            print("s5 error calc: division by 0, returning nan")
            err = nan
        return err
    
    q_squared_lower_bound = 0
    q_squared_upper_bound = 20

    dataframe = dataframe[
        (dataframe[Names_of_Variables().q_squared]>q_squared_lower_bound) 
        & (dataframe[Names_of_Variables().q_squared]<q_squared_upper_bound)
    ]

    groupby_binned = bin_data(
        dataframe=dataframe, 
        name_of_binning_variable=Names_of_Variables().q_squared, 
        start=q_squared_lower_bound,
        stop=q_squared_upper_bound,
        num_bins=num_bins,     
    )
    
    s5s = groupby_binned.apply(calc_s5)
    errs = groupby_binned.apply(calc_s5_err)

    bin_middles = calc_bin_middles(
        start=q_squared_lower_bound, 
        stop=q_squared_upper_bound, 
        num_bins=num_bins,
    )

    return bin_middles, s5s, errs


def calc_s5_of_q_squared_over_delta_C9_values(dataframe, num_bins):

    delta_C9_values = []
    bin_middles_over_delta_C9_values = []
    s5s_over_delta_C9_values = []
    s5_errors_over_delta_C9_values = []

    for delta_C9_value, subset_dataframe in (dataframe.groupby(Names_of_Labels().unbinned)):

        delta_C9_values.append(delta_C9_value)
        bin_middles, s5s, s5_errs = calc_s5_of_q_squared(
            dataframe=subset_dataframe, 
            num_bins=num_bins,
        )
        bin_middles_over_delta_C9_values.append(bin_middles)
        s5s_over_delta_C9_values.append(s5s)
        s5_errors_over_delta_C9_values.append(s5_errs)

    S5_Results_Over_Delta_C9_Values = namedtuple(
        'S5_Results_Over_Delta_C9_Values',
        [
            'delta_C9_values',
            'bin_middles_over_delta_C9_values',
            's5s_over_delta_C9_values',
            's5_errors_over_delta_C9_values'
        ] 
    )
    
    results = S5_Results_Over_Delta_C9_Values(
        delta_C9_values=delta_C9_values,
        bin_middles_over_delta_C9_values=bin_middles_over_delta_C9_values,
        s5s_over_delta_C9_values=s5s_over_delta_C9_values,
        s5_errors_over_delta_C9_values=s5_errors_over_delta_C9_values
    )
    return results