import pytest
import pandas as pd
from pandas.testing import assert_frame_equal  # To compare DataFrames

from vlib.domain.feature_extractor import extract_date_features


def test_extract_date_features_basic():
    # Create a basic DataFrame with datetime values
    df = pd.DataFrame({
        "datetime": ["2024-11-20 14:30:00", "2023-01-01 08:15:00", "2022-07-15 22:45:00"],
        "value": [100, 200, 300]
    })

    # Expected output DataFrame
    expected_df = pd.DataFrame({
        "datetime": pd.to_datetime(["2024-11-20 14:30:00", "2023-01-01 08:15:00", "2022-07-15 22:45:00"]),
        "value": [100, 200, 300],
        "hour": pd.Series([14, 8, 22], dtype="int32"),
        "year": pd.Series([2024, 2023, 2022], dtype="int32"),
        "weekday": pd.Series([2, 6, 4], dtype="int32"),  # Wednesday, Sunday, Friday
        "month": pd.Series([11, 1, 7], dtype="int32"),
        "date": pd.to_datetime(["2024-11-20", "2023-01-01", "2022-07-15"]).date,
        "day": pd.Series([20, 1, 15], dtype="int32"),
    })

    # Run the function
    result_df = extract_date_features(df)

    # Assert the result matches the expected DataFrame
    assert_frame_equal(result_df, expected_df)


def test_extract_date_features_empty_df():
    # Empty DataFrame input
    df = pd.DataFrame({"datetime": []})

    # Expected output should also be an empty DataFrame with the correct columns
    expected_df = pd.DataFrame({
        "datetime": pd.to_datetime([]),
        "hour": pd.Series([], dtype="int32"),
        "year": pd.Series([], dtype="int32"),
        "weekday": pd.Series([], dtype="int32"),
        "month": pd.Series([], dtype="int32"),
        "date": pd.Series([], dtype="object"),
        "day": pd.Series([], dtype="int32"),
    })

    # Run the function
    result_df = extract_date_features(df)

    # Assert the result matches the expected DataFrame
    assert_frame_equal(result_df, expected_df)


def test_extract_date_features_invalid_datetime():
    # Test with invalid datetime values
    df = pd.DataFrame({
        "datetime": ["invalid_date", "2023-01-01 08:15:00", None],
        "value": [100, 200, 300]
    })

    # Expect the function to raise an error during conversion
    with pytest.raises(Exception):
        extract_date_features(df)


def test_extract_date_features_no_side_effects():
    # Original DataFrame
    df = pd.DataFrame({
        "datetime": ["2024-11-20 14:30:00", "2023-01-01 08:15:00"],
        "value": [100, 200]
    })

    # Copy of the original for comparison
    original_df = df.copy()

    # Run the function
    extract_date_features(df)

    # Ensure the original DataFrame remains unchanged
    assert_frame_equal(df, original_df)
