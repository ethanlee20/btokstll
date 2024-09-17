
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
    sets : list of pd.DataFrame
        The sampled sets.
    labels : list of float
        The corresponding labels.
    """
    
    df_grouped = df.groupby(label)
    
    sets = []
    labels = []

    for label_value, df_label in df_grouped:

        for i in range(m):
            
            df_set = df_label.sample(n=n, replace=True)
            sets.append(df_set)
            labels.append(label_value)

    return sets, labels




