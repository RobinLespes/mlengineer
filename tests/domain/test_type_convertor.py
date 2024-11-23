import pytest
import pandas as pd
from pandas.testing import assert_frame_equal

from vlib.domain.type_convertor import convert_features_to_float


# Import the function to test
# from your_module import convert_features_to_float

def test_convert_features_to_float_basic():
    # Input DataFrame with numeric data
    df = pd.DataFrame({
        "col1": [1, 2, 3],
        "col2": [4.5, 5.5, 6.5]
    })

    # Expected output
    expected_df = pd.DataFrame({
        "col1": [1.0, 2.0, 3.0],
        "col2": [4.5, 5.5, 6.5]
    })

    # Run the function
    result_df = convert_features_to_float(df)

    # Assert the result matches the expected output
    assert_frame_equal(result_df, expected_df)


def test_convert_features_to_float_empty_df():
    # Input empty DataFrame
    df = pd.DataFrame()

    # Expected output: also an empty DataFrame
    expected_df = pd.DataFrame()

    # Run the function
    result_df = convert_features_to_float(df)

    # Assert the result matches the expected output
    assert_frame_equal(result_df, expected_df)


def test_convert_features_to_float_non_numeric():
    # Input DataFrame with non-numeric data
    df = pd.DataFrame({
        "col1": ["a", "b", "c"],
        "col2": [1, 2, 3]
    })

    # Expect the function to raise an error due to non-numeric values
    with pytest.raises(ValueError):
        convert_features_to_float(df)


def test_convert_features_to_float_no_side_effects():
    # Input DataFrame
    df = pd.DataFrame({
        "col1": [1, 2, 3],
        "col2": [4.5, 5.5, 6.5]
    })

    # Make a copy of the original DataFrame for comparison
    original_df = df.copy()

    # Run the function
    convert_features_to_float(df)

    # Assert the original DataFrame is unchanged
    assert_frame_equal(df, original_df)


def test_convert_features_to_float_mixed_numeric_formats():
    # Input DataFrame with integers, floats, and strings that represent numbers
    df = pd.DataFrame({
        "col1": [1, "2", 3.0],
        "col2": ["4.5", 5.5, 6]
    })

    # Expected output with all columns converted to floats
    expected_df = pd.DataFrame({
        "col1": [1.0, 2.0, 3.0],
        "col2": [4.5, 5.5, 6.0]
    })

    # Run the function
    result_df = convert_features_to_float(df)

    # Assert the result matches the expected output
    assert_frame_equal(result_df, expected_df)
