import argparse
import concurrent.futures
import csv
import itertools
from concurrent.futures import as_completed
from functools import lru_cache

import geonamescache
import numpy as np
from geopy.distance import geodesic
from tqdm import tqdm

MAX_DISTANCE = 20_037.5
CACHE = geonamescache.GeonamesCache()


# Add argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--country", help="Specify the country code", type=str, default="US"
    )
    parser.add_argument(
        "-w", "--workers", help="Specify the number of workers", type=int, default=1
    )
    parser.add_argument(
        "-s",
        "--chunk-size",
        help="Specify chunk size for batching calculations",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "-o",
        "--output-file",
        help="Specify the name of the output file (file.csv)",
        type=str,
        default="distances.csv",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Option to shuffle combinations list before iterating over it",
    )
    args = parser.parse_args()
    return args


@lru_cache(maxsize=None)
def get_coordinates(city_name, country_code="US"):
    """
    Get the coordinates of a city.

    Parameters
    ----------
    city_name : str
        The name of the city.
    country_code : str, optional
        The country code of the city, by default 'US'.

    Returns
    -------
    tuple
        A tuple containing the latitude and longitude of the city,
        or None if the city is not found.
    """
    city = find_city(city_name, country_code)
    if city is None:
        return None
    return city.get("latitude"), city.get("longitude")


@lru_cache(maxsize=None)
def find_city(city_name, country_code="US"):
    """
    Finds the matching city.

    Parameters
    ----------
    city_name : str
        The name of the city.
    country_code : str, optional
        The country code of the city, by default 'US'.

    Returns
    -------
    city
        A dict containing the raw data about the city.

    """
    search_results = CACHE.get_cities_by_name(city_name)
    # search_results = [
    #     list(c.values())[0] for c in search_results
    # ]
    search_results = [inner_dict for d in search_results for inner_dict in d.values()]
    if not search_results:  # if not found by name, search alternatenames
        search_results = CACHE.search_cities(
            city_name, attribute="alternatenames", case_sensitive=True
        )
    # filter search results to match requested country
    # and avoid wasted computation if coordinates missing
    search_results = [
        d
        for d in search_results
        if (d.get("countrycode") == country_code) & (d.get("longitude") is not None)
    ]

    if not search_results:
        return None
    populations = [city.get("population") for city in search_results]
    city = search_results[np.argmax(populations)]
    return city


def get_distance(city1, city2, country1="US", country2="US"):
    """
    Get the distance between two cities in kilometers.

    Parameters
    ----------
    city1 : str
        The name of the first city.
    city2 : str
        The name of the second city.
    country1 : str, optional
        The country code of the first city, by default 'US'.
    country2 : str, optional
        The country code of the second city, by default 'US'.

    Returns
    -------
    float
        The distance between the two cities in kilometers,
        or None if one or both city names were not found.
    """
    city1_coords = get_coordinates(city1, country1)
    city2_coords = get_coordinates(city2, country2)

    if (city1_coords is None) or (city2_coords is None):
        return None

    return geodesic(city1_coords, city2_coords).km


def calculate_distance(pair):
    city1, city2 = pair
    distance = get_distance(city1, city2)
    return city1, city2, distance


def main(args):
    output_file = args.output_file
    shuffle = args.shuffle
    country_code = args.country
    chunk_size = args.chunk_size
    max_workers = args.workers

    cities = CACHE.get_cities()
    us_cities = {
        k: c
        for k, c in cities.items()
        if (c.get("countrycode") == country_code) & (c.get("longitude") is not None)
    }
    # & (c.get("population", 0) > 5e4)

    cities = list(us_cities.values())
    unique_names = set([c.get("name") for c in cities])
    unique_names = sorted(list(unique_names))
    # unique_cities = [c for c in cities if c.get("name") in unique_names]
    print(f"Num cities: {len(cities)}, unique names: {len(unique_names)}")
    city_combinations = list(itertools.combinations(unique_names, 2))
    if shuffle:
        np.random.shuffle(city_combinations)

    # chunk size, city_combinations, max_workers, output_file
    num_chunks = len(city_combinations) // chunk_size + 1
    with open(output_file, "w", newline="") as csvfile:
        fieldnames = ["city_from", "city_to", "distance"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        try:
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
            for i in tqdm(
                range(num_chunks),
                total=num_chunks,
                desc="Processing chunks",
                ncols=100,
                bar_format="{l_bar}{bar}{r_bar}",
            ):
                chunk = city_combinations[(i * chunk_size) : (i + 1) * chunk_size]
                futures = {
                    executor.submit(calculate_distance, pair): pair for pair in chunk
                }
                for future in as_completed(futures):
                    city_from, city_to, distance = future.result()
                    if distance is not None:
                        writer.writerow(
                            {
                                "city_from": city_from,
                                "city_to": city_to,
                                "distance": distance,
                            }
                        )
                        csvfile.flush()  # write to disk immediately
        except KeyboardInterrupt:
            print("Interrupted. Terminating processes...")
            executor.shutdown(wait=False)
            raise SystemExit("Execution terminated by user.")

        print(f"Wrote {output_file}")


if __name__ == "__main__":
    # preliminary check
    assert find_city("New York City") is not None
    assert find_city("NYC") is not None
    assert round(get_distance("NYC", "Jamaica"), 2) == 17.11
    args = parse_args()
    main(args)
    # perform check
    print("Performing a quick validation...")
    import pandas as pd

    df = pd.read_csv(args.output_file)
    assert df["distance"].min() > 0
    assert df["distance"].max() < MAX_DISTANCE
