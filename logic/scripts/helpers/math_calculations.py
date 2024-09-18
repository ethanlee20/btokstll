
"""
Functions for dealing with dataframes of vectors and matrices.

Note: 
Vector (matrix) dataframes contain one vector (matrix) per row. 
The elements of a row are the components of the vector (matrix).
"""


import numpy as np
import pandas as pd


def square_matrix_transform(df_matrix, df_vec):
    """
    Multiply a vector dataframe by a square matrix dataframe.

    Return a transformed vector dataframe.
    """

    assert np.sqrt(df_matrix.shape[1]) == df_vec.shape[1]
    dims = df_vec.shape[1]

    result = pd.DataFrame(
        data=np.zeros(shape=df_vec.shape),
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
    Compute the dot products of two vector dataframes.

    Return a series of the results (computed row-wise).
    """

    assert df_vec1.shape[1] == df_vec2.shape[1]
    dims = df_vec1.shape[1]

    result = pd.Series(
        data=np.zeros(len(df_vec1)),
        index=df_vec1.index,
        dtype="float64",
    )
    for dim in range(dims):
        result += (
            df_vec1.iloc[:, dim] * df_vec2.iloc[:, dim]
        )
    return result


def vector_magnitude(df_vec):
    """
    Compute the magnitude of each vector in a vector dataframe.

    Return a series of the magnitudes.
    """

    return np.sqrt(dot_product(df_vec, df_vec))


def cosine_angle(df_vec1, df_vec2):
    """
    Find the cosine of the angle between vectors in vector dataframes.

    Return a series of the results (computed row-wise).
    """

    return dot_product(df_vec1, df_vec2) / (
        vector_magnitude(df_vec1)
        * vector_magnitude(df_vec2)
    )


def cross_product_3d(df_3vec1, df_3vec2):
    """
    Find the cross product of vectors in two 3-dimentional vector dataframes.

    Return vector dataframe of results (computed row-wise).
    """

    assert df_3vec1.shape[1] == df_3vec2.shape[1] == 3
    assert df_3vec1.shape[0] == df_3vec2.shape[0]
    assert df_3vec1.index.equals(df_3vec2.index)

    def clean(df_3vec):
        df_3vec = df_3vec.copy()
        df_3vec.columns = ["x", "y", "z"]
        return df_3vec

    df_3vec1 = clean(df_3vec1)
    df_3vec2 = clean(df_3vec2)

    result = pd.DataFrame(
        data=np.zeros(shape=df_3vec1.shape),
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
    Compute the unit normal dataframe of planes specified by two vector dataframes.

    Return a vector dataframe of the results.
    """

    df_normal_vec = cross_product_3d(df_3vec1, df_3vec2)
    df_normal_unit_vec = df_normal_vec.divide(
        vector_magnitude(df_normal_vec), axis="index"
    )

    return df_normal_unit_vec