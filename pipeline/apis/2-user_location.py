#!/usr/bin/env python3
"""A script that prints the location of a specific user"""


import requests


def get_location(url):
    """A function that prints the location of a specific user"""
    r = requests.get(url)

    if r.status_code == 200:
        data = r.json()
        return data["location"]

    elif r.status_code == 404:
        return "Not found"

    elif r.status_code == 403:
        return f"Reset in {r.headers.get('X-Ratelimit-Reset')} sec"


if __name__ == '__main__':
    url = input("Enter GitHub API URL: ")
    location = get_location(url)
    print(location)
