import pandas as pd

from utils.cleaning import clean_string_columns, group_and_clean_spreadsheets, split_words
from utils.geo_utils import dms_to_decimal, haversine_distance, parse_dms_coordinate


def test_clean_string_columns_basic():
    df = pd.DataFrame({"city": [" New York! ", "Los Angeles?"], "name": ["A_B", "C D"]})
    out = clean_string_columns(df, ["city", "name"], lowercase=True, keep_underscore=True)
    assert out["city"].tolist() == ["new york", "los angeles"]
    assert out["name"].tolist() == ["a_b", "c d"]


def test_split_words_basic():
    df = pd.DataFrame({"text": ["alpha beta", "gamma"]})
    out = split_words(df, "text", "words")
    assert out["words"].tolist() == [["alpha", "beta"], ["gamma"]]


def test_group_and_clean_spreadsheets_with_dataframes():
    a = pd.DataFrame({"group": ["alpha", "alpha"], "value": ["New York!", "Boston?"]})
    b = pd.DataFrame({"group": ["beta"], "value": ["Chicago!"]})
    out = group_and_clean_spreadsheets([a, b], columns=["value"], group_by="group")
    assert out["group"].tolist() == ["alpha", "beta"]
    assert out["count"].tolist() == [2, 1]


def test_geo_parse_and_distance():
    lat, lon = parse_dms_coordinate("39°44'24.7\"N 104°51'14.8\"W")
    assert abs(lat - 39.70686111111111) < 1e-9
    assert abs(lon + 104.85355555555556) < 1e-9
    assert abs(dms_to_decimal(1, 30, 0, "S") + 1.5) < 1e-9
    assert haversine_distance(0, 0, 0, 1) > 0
