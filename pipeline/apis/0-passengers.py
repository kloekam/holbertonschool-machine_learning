#!/usr/bin/env python3
"""
A script that creates a method
that returns the list of ships that can hold a given number
of passengers
"""


import requests


def availableShips(passengerCount):
    """
    A function that returns the list of ships
    that can hold a given number
    of passengers
    """
    url = 'https://swapi-api.hbtn.io/api/starships/'
    ships = []
    while url is not None:
        r = requests.get(url)
        data = r.json()
        ships.extend(data["results"])
        url = data["next"]

    availableShips = []
    for availableShip in ships:
        passengers = (
            availableShip["passengers"]
            .replace(",", "")
            .replace("n/a", "0")
            .replace("unknown", "0")
            )
        if passengers.isdigit() and int(passengers) >= passengerCount:
            availableShips.append(availableShip["name"])

    return availableShips
