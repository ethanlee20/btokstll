"""
Pandas dataframe / series math
"""


def square_matrix_transform(df_matrix, df_vec):

    """
    Multiply a dataframe of vectors 
    by a dataframe of square matrices.

    Only works for square matrices.
   
    Parameters
    ----------
    df_matrix : pandas.DataFrame
        Dataframe of matrices.
    df_vec : pandas.DataFrame
        Dataframe of vectors.

    Returns
    -------
    result : pandas.DataFrame
        Transformed vector dataframe.
    """

    if not (
        numpy.sqrt(df_matrix.shape[1]) 
        == df_vec.shape[1]
    ):
        raise ValueError("Matrix must be square.")

    dims = df_vec.shape[1]

    result = pandas.DataFrame(
        data=numpy.zeros(shape=df_vec.shape),
        index=df_vec.index,
        columns=df_vec.columns,
        dtype="float64",
    )

    for i in range(dims):
        for j in range(dims):
            result.iloc[:, i] += (
                df_matrix.iloc[:, dims * i + j]
                * df_vec.iloc[:, j]
            )
    return result


def dot_product(df_vec1, df_vec2):
    """
    Compute the dot products of two 
    vector dataframes.
    Computed row-wise.
    
    Parameters
    ----------
    df_vec1 : pandas.DataFrame
        Dataframe of vectors
    df_vec2 : pandas.DataFrame
        Dataframe of vectors
    Returns
    -------
    result : pandas.Series
        A series of the results.
    """

    if not (
        df_vec1.shape[1] == df_vec2.shape[1]
    ):
        raise ValueError(
            "Vector dimensions do not match."
        )
    dims = df_vec1.shape[1]

    result = pandas.Series(
        data=numpy.zeros(len(df_vec1)),
        index=df_vec1.index,
        dtype="float64",
    )
    for dim in range(dims):
        result += (
            df_vec1.iloc[:, dim] 
            * df_vec2.iloc[:, dim]
        )
    return result


def vector_magnitude(df_vec):
    
    """
    Compute the magnitude of each vector 
    in a vector dataframe.

    Parameters
    ----------
    df_vec : pandas.DataFrame
        Dataframe of vectors.

    Returns
    -------
    result : pandas.Series
        Series of the magnitudes.
    """

    result = numpy.sqrt(dot_product(df_vec, df_vec))
    return result


def cosine_angle(df_vec1, df_vec2):
    """
    Find the cosine of the angle 
    between vectors in vector dataframes.
    Computed row-wise.
    
    Parameters
    ----------
    df_vec1 : pandas.DataFrame
        Dataframe of vectors.
    df_vec2 : pandas.DataFrame
        Dataframe of vectors.

    Returns
    -------
    pandas.Series
        A series of the results.
    """

    result = dot_product(df_vec1, df_vec2) / (
        vector_magnitude(df_vec1)
        * vector_magnitude(df_vec2)
    )

    return result


def cross_product_3d(df_3vec1, df_3vec2):

    """
    Find the cross product of vectors of two 
    3-dimensional vector dataframes.
    Computed row-wise.

    Parameters
    ----------
    df_3vec1 : pandas.DataFrame
        Dataframe of 3-dimensional vectors.
    df_3vec2 : pandas.DataFrame
        Dataframe of 3-dimensional vectors.

    Returns
    -------
    result : pandas.DataFrame 
        Dataframe of vectors.
    """

    assert (
        df_3vec1.shape[1] 
        == df_3vec2.shape[1] 
        == 3
    )
    assert df_3vec1.shape[0] == df_3vec2.shape[0]
    assert df_3vec1.index.equals(df_3vec2.index)

    def clean(df_3vec):
        df_3vec = df_3vec.copy()
        df_3vec.columns = ["x", "y", "z"]
        return df_3vec

    df_3vec1 = clean(df_3vec1)
    df_3vec2 = clean(df_3vec2)

    result = pandas.DataFrame(
        data=numpy.zeros(shape=df_3vec1.shape),
        index=df_3vec1.index,
        columns=df_3vec1.columns,
        dtype="float64",
    )

    result["x"] = (
        df_3vec1["y"] * df_3vec2["z"]
        - df_3vec1["z"] * df_3vec2["y"]
    )
    result["y"] = (
        df_3vec1["z"] * df_3vec2["x"]
        - df_3vec1["x"] * df_3vec2["z"]
    )
    result["z"] = (
        df_3vec1["x"] * df_3vec2["y"]
        - df_3vec1["y"] * df_3vec2["x"]
    )

    return result


def unit_normal(df_3vec1, df_3vec2):
    
    """
    Compute the unit normal dataframe of 
    planes specified by two vector dataframes.

    Parameters
    ----------
    df_3vec1 : pandas.DataFrame
        Dataframe of 3 dimensional vectors.
    df_3vec2 : pandas.DataFrame
        Dataframe of 3 dimensional vectors.
    
    Returns
    -------
    df_unit_normal_vec : pandas.DataFrame
        Dataframe of vectors.
    """

    df_normal_vec = cross_product_3d(
        df_3vec1, 
        df_3vec2
    )
    df_unit_normal_vec = df_normal_vec.divide(
        vector_magnitude(df_normal_vec), axis="index"
    )

    return df_unit_normal_vec

