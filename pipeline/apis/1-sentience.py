#!/usr/bin/env python3
"""
A script that creates a method that
returns the list of names of the home planets
of all sentient species
"""


import requests


def sentientPlanets():
    """
    A method that returns the list of names of the
    home planets of all sentient species
    """
    url = "https://swapi-api.hbtn.io/api/species/"
    planets = []

    while url is not None:
        r = requests.get(url)
        data = r.json()
        planets.extend(data["results"])
        url = data["next"]

    sentientPlanets = []
    for species in planets:
        if "sentient" in species["classification"].lower() or \
             "sentient" in species["designation"].lower():
            home = species["homeworld"]
            if home:
                planet_data = requests.get(home).json()
                sentientPlanets.append(planet_data["name"])

    return sentientPlanets
