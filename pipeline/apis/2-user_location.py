#!/usr/bin/env python3
"""A script that prints the location of a specific user"""


import requests
import sys
from datetime import datetime


def get_location(url):
    """A function that prints the location of a specific user"""
    r = requests.get(url)

    if r.status_code == 200:
        data = r.json()
        return data["location"]

    elif r.status_code == 404:
        return "Not found"

    elif r.status_code == 403:
        reset_time = int(r.headers.get('X-Ratelimit-Reset'))
        current_time = int(datetime.now().timestamp())
        mins = (reset_time - current_time) // 60
        return f"Reset in {mins} min"


if __name__ == '__main__':
    url = sys.argv[1]
    location = get_location(url)
    print(location)
