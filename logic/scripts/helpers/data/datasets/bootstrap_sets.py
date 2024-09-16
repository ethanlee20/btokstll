
"""A function for bootstrapping sets."""


def bootstrap_sets(df, label, n, m):
    """
    Bootstrap sets from the distribution of each label in a
    source dataframe.

    Bootstrapping refers to sampling with replacement.
    
    Parameters
    ----------
    df : pd.DataFrame
        The source dataframe to sample from
    label : str 
        Name of the column specifying the labels.
    n : int
        The number of elements per set.
    m : int
        The number of sets per label. 
    
    Returns
    -------
    list of pd.DataFrame
        The sampled sets.
    """
    
    df_grouped = df.groupby(label)
    
    result = []
    
    for _, df_label in df_grouped:

        for i in range(m):
            
            df_set = df.sample(n=n, replace=True)
            result.append(df_set)

    return result




