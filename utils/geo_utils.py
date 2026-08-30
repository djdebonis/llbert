from __future__ import annotations

import numpy as np
import regex as re


def dms_to_decimal(degrees, minutes, seconds, direction):
    decimal = degrees + minutes / 60 + seconds / 3600

    if direction.upper() in ["S", "W"]:
        decimal *= -1

    return decimal


def parse_dms_coordinate(coord_string):
    """Parse a DMS coordinate such as 39°44'24.7"N 104°51'14.8"W."""
    pattern = r"""(
        \d+)[°]\s*
        (\d+)'\s*
        ([\d\.]+)"\s*
        ([NSEW])
    """

    matches = re.findall(pattern, coord_string, re.VERBOSE)

    if len(matches) != 2:
        raise ValueError("Could not parse coordinate string")

    lat_match = matches[0]
    lon_match = matches[1]

    lat = dms_to_decimal(
        float(lat_match[0]),
        float(lat_match[1]),
        float(lat_match[2]),
        lat_match[3],
    )

    lon = dms_to_decimal(
        float(lon_match[0]),
        float(lon_match[1]),
        float(lon_match[2]),
        lon_match[3],
    )

    return lat, lon


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points in kilometers."""
    r = 6371

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))
    return r * c


def greet():
    print("hello from geo_utils")
