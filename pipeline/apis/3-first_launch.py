#!/usr/bin/env python3
"""A script that displays the first launch with some details"""


import requests


def launch():
    """
    A function that displays the first launch with the information of:
    Name of the launch
    The date (in local time)
    The rocket name
    The name (with the locality) of the launchpad
    """
    r = requests.get("https://api.spacexdata.com/v4/launches")
    launcehs_data = r.json()

    if not launcehs_data:
        print("No launches found")
        return
    sorted_launches = sorted(launcehs_data, key=lambda x: x["date_unix"])[0]

    rocket_id = sorted_launches["rocket"]
    r = requests.get(
        f"https://api.spacexdata.com/v4/rockets/{rocket_id}")
    rocket_data = r.json()
    rocket_name = rocket_data["name"]

    launchpad_id = sorted_launches["launchpad"]
    r = requests.get(
        f"https://api.spacexdata.com/v4/launchpads/{launchpad_id}")
    launchpad_data = r.json()
    launchpad_name = launchpad_data["name"]
    launchpad_locality = launchpad_data["locality"]

    print(
        f"{sorted_launches['name']} ({sorted_launches['date_local']}) "
        f"{rocket_name} - {launchpad_name} ({launchpad_locality})")


if __name__ == "__main__":
    launch()
