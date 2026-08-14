import geonamescache
from geopy.distance import geodesic

gc = geonamescache.GeonamesCache()
cities = gc.get_cities()
us_cities = {k: c for k, c in cities.items() if c.get("countrycode") == "US"}

print(gc.search_cities("Jamaica"), "\n")
print(gc.search_cities("Manhattan"), "\n")
print("lengths:", len(cities), len(us_cities))


def get_coordinates(city_name, country_code="US"):
    search_results = gc.search_cities(city_name, case_sensitive=True)
    for city in search_results:
        print(f"searching {city}")
        possible_matches = city.get("alternatenames") + [city_name]
        if city_name in possible_matches and city.get("countrycode") == country_code:
            return city.get("latitude"), city.get("longitude")
    return None


def get_distance(city1, city2, country1="US", country2="US"):
    city1_coords = get_coordinates(city1, country1)
    city2_coords = get_coordinates(city2, country2)

    if (city1_coords is None) or (city2_coords is None):
        return None

    return geodesic(city1_coords, city2_coords).km


MAX_DISTANCE = 20_037.5

city1 = "New York"
city2 = "Jamaica"
country1 = "US"
country2 = "US"

distance = get_distance(city1, city2, country1, country2)

if distance is not None:
    print(f"Distance between {city1} and {city2} is {distance:.2f} km.")
else:
    print("One or both city names were not found.")
