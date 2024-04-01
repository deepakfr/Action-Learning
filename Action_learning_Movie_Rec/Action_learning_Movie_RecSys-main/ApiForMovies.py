import json
import requests

def get_images_from_movie_name(movie_name, api_key):
    response = requests.get(
        "https://api.themoviedb.org/3/search/movie?api_key={}&query={}".format(api_key, movie_name)
    )
    if response.status_code == 200:
        data = json.loads(response.content)
        results = data["results"]
        if len(results) > 0:
            movie = results[0]
            if "posters" in movie:
                images = movie["posters"]
                return images
            else:
                return None
        else:
            return None
    else:
        return None

movie_name = "Secret Invasion (2023)"
api_key = "e9c3346ca46a25cec9b7804aebfc98d4"

images = get_images_from_movie_name(movie_name, api_key)

if images is not None:
    for image in images:
        print(image["file_path"])
