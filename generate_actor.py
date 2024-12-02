import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from imdb import IMDb

def query_imdb_movie(imdb_id):
    """
    Query IMDb for movie details by IMDb ID.
    """
    try:
        ia = IMDb()
        print(f"[DEBUG] Querying IMDb for movie ID: {imdb_id}")
        movie = ia.get_movie(imdb_id[2:])  # Strip "tt" prefix for IMDbPy
        if movie:
            movie_title = movie.get("title")
            if movie_title:
                print(f"[INFO] {movie_title}: {imdb_id}")
                return imdb_id, movie_title
            else:
                print(f"[WARNING] No title found for IMDb ID: {imdb_id}")
        else:
            print(f"[WARNING] No movie found for IMDb ID: {imdb_id}")
    except Exception as e:
        print(f"[ERROR] Failed to fetch data for IMDb ID {imdb_id}: {e}")
    return imdb_id, None

def query_imdb_actor(imdb_id):
    """
    Query IMDb for actor details by IMDb ID.
    """
    try:
        ia = IMDb()
        print(f"[DEBUG] Querying IMDb for actor ID: {imdb_id}")
        person = ia.get_person(imdb_id[2:])  # Strip "nm" prefix for IMDbPy
        if person:
            actor_name = person.get("name")
            if actor_name:
                print(f"[INFO] {actor_name}: {imdb_id}")
                return imdb_id, actor_name
            else:
                print(f"[WARNING] No name found for IMDb ID: {imdb_id}")
        else:
            print(f"[WARNING] No person found for IMDb ID: {imdb_id}")
    except Exception as e:
        print(f"[ERROR] Failed to fetch data for IMDb ID {imdb_id}: {e}")
    return imdb_id, None

def generate_mappings_from_images(json_file, actor_output_file="actor_imdb_mapping.json", movie_output_file="movie_imdb_mapping.json", threads=10, limit=20000):
    """
    Generate mappings of actor names and movie titles to IMDb IDs using the images.json file.
    Limit the number of actors and movies processed to a specified limit.
    """
    try:
        # Load images.json
        with open(json_file, "r") as f:
            images_data = json.load(f)
        print(f"[INFO] Loaded {json_file}")
    except FileNotFoundError:
        print(f"[ERROR] File not found: {json_file}")
        return

    # Collect all unique IMDb IDs from 'cast' and 'movie' fields
    actor_ids = set()
    movie_ids = set()
    for item in images_data:
        if "cast" in item:
            actor_ids.update(item["cast"])
        if "movie" in item:
            movie_ids.update(item["movie"])

    print(f"[INFO] Found {len(actor_ids)} unique actor IMDb IDs.")
    print(f"[INFO] Found {len(movie_ids)} unique movie IMDb IDs.")

    # Limit the number of actors and movies to the specified limit
    actor_ids = set(list(actor_ids)[:limit])
    movie_ids = set(list(movie_ids)[:limit])

    print(f"[INFO] Limiting to {len(actor_ids)} actors and {len(movie_ids)} movies.")

    # Load existing mappings to skip already processed IDs
    try:
        with open(actor_output_file, "r") as f:
            actor_mapping = json.load(f)
        print(f"[INFO] Loaded existing actor mappings from {actor_output_file}")
    except (FileNotFoundError, json.JSONDecodeError):
        actor_mapping = {}

    try:
        with open(movie_output_file, "r") as f:
            movie_mapping = json.load(f)
        print(f"[INFO] Loaded existing movie mappings from {movie_output_file}")
    except (FileNotFoundError, json.JSONDecodeError):
        movie_mapping = {}

    # Skip already processed IMDb IDs
    remaining_actor_ids = actor_ids - set(actor_mapping.values())
    remaining_movie_ids = movie_ids - set(movie_mapping.values())
    print(f"[INFO] Processing {len(remaining_actor_ids)} remaining actor IMDb IDs.")
    print(f"[INFO] Processing {len(remaining_movie_ids)} remaining movie IMDb IDs.")

    # Process actors and movies concurrently
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=threads) as executor:
        actor_futures = {executor.submit(query_imdb_actor, imdb_id): imdb_id for imdb_id in remaining_actor_ids}
        movie_futures = {executor.submit(query_imdb_movie, imdb_id): imdb_id for imdb_id in remaining_movie_ids}
        
        # Process actor results
        processed = 0
        for future in as_completed(actor_futures):
            imdb_id, actor_name = future.result()
            processed += 1
            if actor_name:
                actor_mapping[actor_name] = imdb_id
            if processed % 100 == 0:
                with open(actor_output_file, "w") as f:
                    json.dump(actor_mapping, f, indent=4)
                print(f"[INFO] Saved intermediate actor progress to {actor_output_file}")
        
        # Process movie results
        processed = 0
        for future in as_completed(movie_futures):
            imdb_id, movie_title = future.result()
            processed += 1
            if movie_title:
                movie_mapping[movie_title] = imdb_id
            if processed % 100 == 0:
                with open(movie_output_file, "w") as f:
                    json.dump(movie_mapping, f, indent=4)
                print(f"[INFO] Saved intermediate movie progress to {movie_output_file}")

    # Save the final mappings to JSON files
    if actor_mapping:
        with open(actor_output_file, "w") as f:
            json.dump(actor_mapping, f, indent=4)
        print(f"[INFO] Actor-IMDb mapping saved to {actor_output_file}")
    else:
        print("[ERROR] No actor mappings generated.")

    if movie_mapping:
        with open(movie_output_file, "w") as f:
            json.dump(movie_mapping, f, indent=4)
        print(f"[INFO] Movie-IMDb mapping saved to {movie_output_file}")
    else:
        print("[ERROR] No movie mappings generated.")


if __name__ == "__main__":
    # Path to your images.json file
    json_file = "images.json"

    # Generate the mappings
    generate_mappings_from_images(json_file)
