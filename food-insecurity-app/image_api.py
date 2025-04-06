from serpapi import GoogleSearch

SERPAPI_KEY = "6c42aa0ac5cf50f0059651cdfeb83106f5e7f5385052e57b20c667e432e26a38"  

def get_recipe_image(title):
    params = {
        "q": title + " recipe",
        "tbm": "isch",  # image search
        "api_key": SERPAPI_KEY
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    images = results.get("images_results", [])
    if images:
        return images[0]["original"]
    return None
