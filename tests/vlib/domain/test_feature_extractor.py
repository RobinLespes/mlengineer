import pandas as pd
from src.vlib.domain.feature_extractor import extract_date_features


def test_extract_date_features_should_extract_year_from_datetime():
    # Given
    df = pd.DataFrame({"datetime": ["2012-01-01 12:56:00"]})
    expected_year = 2012
    # When
    res = extract_date_features(df)

    # Then
    assert res["year"].values[0] == expected_year