import json
import os

FAV_FILE = "favourites.json"

def load_favourites():
    if not os.path.exists(FAV_FILE):
        with open(FAV_FILE, "w") as f:
            f.write("[]")

    with open(FAV_FILE, "r") as f:
        content = f.read().strip()
        if content:
            return json.loads(content)
        return []

def save_favourites(fav_list):
    with open(FAV_FILE, "w") as f:
        json.dump(fav_list, f, indent=2)
