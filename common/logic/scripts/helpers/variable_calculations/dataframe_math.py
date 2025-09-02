
import numpy
import pandas


def square_matrix_transform(matrix_dataframe, vector_dataframe):

    """
    Multiply a dataframe of vectors 
    by a dataframe of square matrices.
    Return a dataframe.

    Only works for square matrices.
    """

    if not (
        numpy.sqrt(matrix_dataframe.shape[1]) 
        == vector_dataframe.shape[1]
    ):
        raise ValueError("Matrix must be square.")

    vector_length = vector_dataframe.shape[1]

    transformed_vector_dataframe = pandas.DataFrame(
        data=numpy.zeros(shape=vector_dataframe.shape),
        index=vector_dataframe.index,
        columns=vector_dataframe.columns,
        dtype="float64",
    )

    for i in range(vector_length):
        for j in range(vector_length):
            transformed_vector_dataframe.iloc[:, i] += (
                matrix_dataframe.iloc[:, vector_length * i + j]
                * vector_dataframe.iloc[:, j]
            )

    return transformed_vector_dataframe


def dot_product(vector_dataframe_1, vector_dataframe_2):

    """
    Compute the dot products of two vector dataframes.
    """

    if not (vector_dataframe_1.shape[1] == vector_dataframe_2.shape[1]):
        raise ValueError("Vector dimensions do not match.")
    
    vector_length = vector_dataframe_1.shape[1]

    result_series = pandas.Series(
        data=numpy.zeros(len(vector_dataframe_1)),
        index=vector_dataframe_1.index,
        dtype="float64",
    )

    for dimension in range(vector_length):
        result_series += (
            vector_dataframe_1.iloc[:, dimension] 
            * vector_dataframe_2.iloc[:, dimension]
        )

    return result_series


def vector_magnitude(vector_dataframe):
    
    """
    Compute the magnitude of each vector in a vector dataframe.
    Return a series.
    """

    result_series = numpy.sqrt(dot_product(vector_dataframe, vector_dataframe))
    
    return result_series


def cosine_angle(vector_dataframe_1, vector_dataframe_2):
    
    """
    Find the cosine of the angle between vectors in vector dataframes.
    Return a series.
    """

    result_series = (
        dot_product(vector_dataframe_1, vector_dataframe_2) 
        / (
            vector_magnitude(vector_dataframe_1)
            * vector_magnitude(vector_dataframe_2)
        )
    )

    return result_series


def cross_product_3d(three_vector_dataframe_1, three_vector_dataframe_2):

    """
    Find the cross product of 3-dimensional vectors 
    from two vector dataframes.
    Return a vector dataframe.
    """

    assert (
        three_vector_dataframe_1.shape[1] 
        == three_vector_dataframe_2.shape[1] 
        == 3
    )
    assert (
        three_vector_dataframe_1.shape[0] 
        == three_vector_dataframe_2.shape[0]
    )
    assert (
        three_vector_dataframe_1.index.equals(
            three_vector_dataframe_2.index
        )
    )

    three_vector_dataframe_1 = three_vector_dataframe_1.copy()
    three_vector_dataframe_2 = three_vector_dataframe_2.copy()

    three_vector_dataframe_1.columns = ["x", "y", "z"]
    three_vector_dataframe_2.columns = ["x", "y", "z"]

    cross_product_dataframe = pandas.DataFrame(
        data=numpy.zeros(
            shape=three_vector_dataframe_1.shape
        ),
        index=three_vector_dataframe_1.index,
        columns=three_vector_dataframe_1.columns,
        dtype="float64"
    )

    cross_product_dataframe["x"] = (
        three_vector_dataframe_1["y"] * three_vector_dataframe_2["z"]
        - three_vector_dataframe_1["z"] * three_vector_dataframe_2["y"]
    )
    cross_product_dataframe["y"] = (
        three_vector_dataframe_1["z"] * three_vector_dataframe_2["x"]
        - three_vector_dataframe_1["x"] * three_vector_dataframe_2["z"]
    )
    cross_product_dataframe["z"] = (
        three_vector_dataframe_1["x"] * three_vector_dataframe_2["y"]
        - three_vector_dataframe_1["y"] * three_vector_dataframe_2["x"]
    )

    return cross_product_dataframe


def unit_normal(three_vector_dataframe_1, three_vector_dataframe_2):
    
    """
    For planes specified by two three-vector dataframes,
    calculate the unit normal vectors.
    Return a vector dataframe.
    """

    normal_vector_dataframe = cross_product_3d(
        three_vector_dataframe_1, 
        three_vector_dataframe_2
    )
    
    unit_normal_vector_dataframe = normal_vector_dataframe.divide(
        vector_magnitude(normal_vector_dataframe), 
        axis="index"
    )

    return unit_normal_vector_dataframe

