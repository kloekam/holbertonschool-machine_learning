#!/usr/bin/env python3
"""A script that displays the number of launches per rocket"""


import requests


def rocket_frequency():
    """A function that displays the number of launches per rocket"""
    r = requests.get("https://api.spacexdata.com/v4/launches")
    launches_data = r.json()

    rocket_count = {}
    for launch in launches_data:
        rocket_id = launch["rocket"]
        rocket_count[rocket_id] = rocket_count.get(rocket_id, 0) + 1

    rocket_names = {}
    for rocket_id in rocket_count.keys():
        r_rocket = requests.get(
            (f"https://api.spacexdata.com/v4/rockets/{rocket_id}")
            )
        rocket_data = r_rocket.json()
        rocket_names[rocket_id] = rocket_data.get("name", "Uknown")

    rocket_list = []
    for rid, count in rocket_count.items():
        rocket_list.append((rocket_names[rid], count))
        rockets_sorted = sorted(rocket_list, key=lambda x: (-x[1], x[0]))

    for rocket_name, count in rockets_sorted:
        print(f"{rocket_name}: {count}")


if __name__ == '__main__':
    rocket_frequency()
