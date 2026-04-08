"""
cli_demo.py — A standalone CLI demo of the ATAI movie chatbot.

The original bot (GoodBot.py / TobyKh_Bot.py) connects to UZH's Speakeasy
chatroom platform and depends on a 7M-triple knowledge graph (14_graph.nt)
that isn't tracked in this repo. This file is a self-contained demo that
mirrors the same query-routing logic and visual style so the bot can be
demonstrated and screen-recorded without the Speakeasy infrastructure.

Run:  python cli_demo.py
"""

import json
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

# --- colorama fallback ------------------------------------------------------
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init()
except ImportError:
    class _Stub:
        def __getattr__(self, _): return ""
    Fore = Style = _Stub()


HERE = os.path.dirname(os.path.abspath(__file__))


# --- Sample knowledge base --------------------------------------------------
# Curated facts so the demo can answer questions accurately even without
# the full RDF graph. Keys are normalised lowercase movie/person names.
FACTS = {
    "the godfather": {
        "director": "Francis Ford Coppola",
        "screenwriter": "Mario Puzo, Francis Ford Coppola",
        "publication date": "1972-03-24",
        "genre": "crime, drama",
        "cast member": "Marlon Brando, Al Pacino, James Caan, Robert Duvall",
        "box office": "$250–291 million",
        "country of origin": "United States",
        "composer": "Nino Rota",
    },
    "pulp fiction": {
        "director": "Quentin Tarantino",
        "screenwriter": "Quentin Tarantino, Roger Avary",
        "publication date": "1994-10-14",
        "genre": "crime, drama",
        "cast member": "John Travolta, Samuel L. Jackson, Uma Thurman, Bruce Willis",
        "box office": "$213.9 million",
        "country of origin": "United States",
    },
    "inception": {
        "director": "Christopher Nolan",
        "screenwriter": "Christopher Nolan",
        "publication date": "2010-07-16",
        "genre": "science fiction, action, thriller",
        "cast member": "Leonardo DiCaprio, Joseph Gordon-Levitt, Elliot Page, Tom Hardy",
        "box office": "$839.1 million",
        "composer": "Hans Zimmer",
    },
    "the dark knight": {
        "director": "Christopher Nolan",
        "screenwriter": "Jonathan Nolan, Christopher Nolan",
        "publication date": "2008-07-18",
        "genre": "superhero, crime, thriller",
        "cast member": "Christian Bale, Heath Ledger, Aaron Eckhart, Maggie Gyllenhaal",
        "box office": "$1.006 billion",
        "composer": "Hans Zimmer, James Newton Howard",
    },
    "forrest gump": {
        "director": "Robert Zemeckis",
        "screenwriter": "Eric Roth",
        "publication date": "1994-07-06",
        "genre": "drama, romance",
        "cast member": "Tom Hanks, Robin Wright, Gary Sinise, Mykelti Williamson",
        "box office": "$678.2 million",
    },
    "the matrix": {
        "director": "Lana Wachowski, Lilly Wachowski",
        "screenwriter": "Lana Wachowski, Lilly Wachowski",
        "publication date": "1999-03-31",
        "genre": "science fiction, action",
        "cast member": "Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss, Hugo Weaving",
    },
    "interstellar": {
        "director": "Christopher Nolan",
        "screenwriter": "Jonathan Nolan, Christopher Nolan",
        "publication date": "2014-11-07",
        "genre": "science fiction, drama",
        "cast member": "Matthew McConaughey, Anne Hathaway, Jessica Chastain, Michael Caine",
        "composer": "Hans Zimmer",
    },
    "the shawshank redemption": {
        "director": "Frank Darabont",
        "screenwriter": "Frank Darabont",
        "publication date": "1994-09-23",
        "genre": "drama",
        "cast member": "Tim Robbins, Morgan Freeman, Bob Gunton, William Sadler",
    },
    "fight club": {
        "director": "David Fincher",
        "screenwriter": "Jim Uhls",
        "publication date": "1999-10-15",
        "genre": "drama, thriller",
        "cast member": "Brad Pitt, Edward Norton, Helena Bonham Carter, Meat Loaf",
    },
}

# Recommendation buckets keyed by genre/style.
RECOMMENDATIONS = {
    "the godfather":   ["Goodfellas", "Once Upon a Time in America", "Heat", "Casino", "The Departed"],
    "pulp fiction":    ["Reservoir Dogs", "Jackie Brown", "Trainspotting", "Snatch", "Lock, Stock and Two Smoking Barrels"],
    "inception":       ["Memento", "The Prestige", "Shutter Island", "Tenet", "Source Code"],
    "the dark knight": ["Batman Begins", "Joker", "Logan", "Watchmen", "Sicario"],
    "the matrix":      ["Dark City", "Equilibrium", "Ghost in the Shell", "Existenz", "Total Recall"],
    "interstellar":    ["Arrival", "Gravity", "Contact", "The Martian", "Ad Astra"],
    "forrest gump":    ["The Curious Case of Benjamin Button", "Cast Away", "The Green Mile", "Big Fish", "A Beautiful Mind"],
}

# Crowd-sourced fact validation stubs (from the real bot's CrowdSource flow).
CROWD = [
    {
        "claim": ("the godfather", "director", "Francis Ford Coppola"),
        "support": 3, "reject": 0, "agreement": 1.00,
    },
    {
        "claim": ("inception", "director", "Steven Spielberg"),
        "support": 0, "reject": 3, "agreement": 1.00,
    },
    {
        "claim": ("pulp fiction", "publication date", "1994"),
        "support": 3, "reject": 1, "agreement": 0.75,
    },
]


# --- Helpers ----------------------------------------------------------------
def slow_print(text, delay=0.012, end="\n"):
    """Type-style print so the GIF feels alive without being painful to watch."""
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write(end)
    sys.stdout.flush()


def load_json_quiet(path):
    try:
        with open(os.path.join(HERE, path), "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


# --- Banner -----------------------------------------------------------------
DUCK = r"""
              ---
           -        --
       --( /     \ )XXXXXXXXXXXXX
   --XXX(   O   O  )XXXXXXXXXXXXXXX-
  /XXX(       U     )        XXXXXXX\
/XXXXX(              )--   XXXXXXXXXXX\
/XXXXX/ (      O     )   XXXXXX   \XXXXX\
XXXXX/   /            XXXXXX   \   \XXXXX----
XXXXXX  /          XXXXXX         \  ----  -
XXX  /          XXXXXX      \           ---
  --  /      /\  XXXXXX            /     ---=
   /    XXXXXX              '--- XXXXXX
--\/XXX\ XXXXXX                      /XXXXX
   \XXXXXXXXX                        /XXXXX/
    \XXXXXX                         /XXXXX/
      \XXXXX--  /                -- XXXX/
       --XXXXXXX---------------  XXXXX--
          \XXXXXXXXXXXXXXXXXXXXXXXX-
            --XXXXXXXXXXXXXXXXXX-
"""


def banner():
    print("=" * 60)
    print(f"{Fore.CYAN}        ATAI Movie Chatbot — CLI Demo{Style.RESET_ALL}")
    print("=" * 60)


def fake_init():
    """Mimic the real bot's startup sequence so the GIF feels authentic."""
    steps = [
        ("Connecting to knowledge graph",   0.30),
        ("Loading entity embeddings",       0.35),
        ("Loading relation embeddings",     0.20),
        ("Initialising NER pipeline",       0.40),
        ("Indexing crowd-source data",      0.25),
        ("Warming up SPARQL engine",        0.20),
    ]
    for label, delay in steps:
        sys.stdout.write(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} {label}...")
        sys.stdout.flush()
        time.sleep(delay)
        print(f" {Fore.GREEN}done{Style.RESET_ALL}")
    print()
    print(f"{Fore.YELLOW}{DUCK}{Style.RESET_ALL}")
    print("╔══════════════════════════════════════════════════════════╗")
    print(f"║                  {Fore.GREEN}BOT IS NOW LIVE!{Style.RESET_ALL}                        ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print("Ask me factual questions about movies, request recommendations,")
    print("or ask to see images. Type 'exit' to quit.\n")


# --- Question routing -------------------------------------------------------
RELATION_KEYWORDS = {
    "director":         ["who directed", "director of", "who is the director"],
    "screenwriter":     ["who wrote", "screenwriter", "who is the writer"],
    "publication date": ["when was", "release date", "released", "when did", "year of"],
    "genre":            ["what genre", "what kind of movie", "type of film"],
    "cast member":      ["who is in", "cast of", "who stars in", "cast members"],
    "box office":       ["how much did", "box office", "revenue"],
    "composer":         ["composed", "score for", "music for", "composer"],
    "country of origin":["country", "where is", "where was"],
}

MULTIMEDIA_TRIGGERS = ["show me", "picture of", "image of", "frames of", "scene of", "look like"]
RECOMMENDATION_TRIGGERS = [
    "recommend", "suggest", "movies like", "films like", "similar to",
    "if i liked", "if you liked", "for a fan of", "i love", "i enjoyed",
]
CROWD_TRIGGERS = ["is it true", "really the", "did ", "actually"]


def find_movie(question_lc):
    """Match the longest known movie title in the question."""
    matches = [t for t in FACTS.keys() if t in question_lc]
    if matches:
        return max(matches, key=len)
    matches = [t for t in RECOMMENDATIONS.keys() if t in question_lc]
    if matches:
        return max(matches, key=len)
    return None


def find_relation(question_lc):
    for rel, phrases in RELATION_KEYWORDS.items():
        if any(p in question_lc for p in phrases):
            return rel
    return None


def title_case(s):
    return " ".join(w.capitalize() for w in s.split())


def answer_factual(movie, relation):
    facts = FACTS.get(movie)
    if not facts:
        return f"I don't have any information on '{title_case(movie)}' yet."
    if relation not in facts:
        available = ", ".join(facts.keys())
        return (f"I know about '{title_case(movie)}' but not its {relation}. "
                f"Try one of: {available}.")
    value = facts[relation]
    return f"The {relation} of {title_case(movie)} is: {value}."


def answer_recommendation(movie):
    recs = RECOMMENDATIONS.get(movie)
    if not recs:
        # fall back to a random pick
        movie = random.choice(list(RECOMMENDATIONS.keys()))
        recs = RECOMMENDATIONS[movie]
        return (f"Based on a similar style to {title_case(movie)}, you might enjoy:\n"
                + "\n".join(f"  • {r}" for r in recs))
    return (f"If you liked {title_case(movie)}, you might also enjoy:\n"
            + "\n".join(f"  • {r}" for r in recs))


POSTER_CACHE = os.path.join(HERE, ".poster_cache")

# Wikipedia page slugs for the movies we know about — required because
# titles need disambiguators (e.g. "Inception_(2010_film)") to hit the
# right article on the first try.
WIKI_SLUGS = {
    "the godfather":             "The_Godfather",
    "pulp fiction":              "Pulp_Fiction",
    "inception":                 "Inception_(2010_film)",
    "the dark knight":           "The_Dark_Knight",
    "forrest gump":              "Forrest_Gump",
    "the matrix":                "The_Matrix",
    "interstellar":              "Interstellar_(film)",
    "the shawshank redemption":  "The_Shawshank_Redemption",
    "fight club":                "Fight_Club",
}


def _fetch_poster(movie):
    """Download the real Wikipedia poster for a movie. Returns a local path."""
    os.makedirs(POSTER_CACHE, exist_ok=True)
    cache_path = os.path.join(POSTER_CACHE, movie.replace(" ", "_") + ".jpg")
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        return cache_path

    slug = WIKI_SLUGS.get(movie, title_case(movie).replace(" ", "_"))
    summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}"
    headers = {"User-Agent": "Chatbot_ATAI-CLI-Demo/1.0 (github.com/wallacees12)"}

    try:
        req = urllib.request.Request(summary_url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
        img_url = (data.get("originalimage", {}).get("source")
                   or data.get("thumbnail", {}).get("source"))
        if not img_url:
            return None
        img_req = urllib.request.Request(img_url, headers=headers)
        with urllib.request.urlopen(img_req, timeout=10) as img_resp:
            with open(cache_path, "wb") as f:
                f.write(img_resp.read())
        return cache_path
    except (urllib.error.URLError, json.JSONDecodeError, KeyError, TimeoutError):
        return None


def _render_inline(image_path):
    """Render an image inline in the terminal via chafa, if available."""
    chafa = shutil.which("chafa")
    if not chafa:
        return False
    try:
        # chafa autodetects the best protocol (Kitty graphics in Ghostty,
        # iTerm2 inline in iTerm, sixel/ansi-blocks elsewhere).
        subprocess.run([chafa, "--size=40x22", image_path], check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def answer_multimedia(movie):
    if not movie:
        return "I couldn't tell which movie you wanted to see."
    title = title_case(movie)

    poster_path = _fetch_poster(movie)
    if poster_path and _render_inline(poster_path):
        return f"[multimedia] Wikipedia poster for {title} (rendered above)"

    if not shutil.which("chafa"):
        return (f"[multimedia] Found a poster for {title} but chafa isn't "
                f"installed. Run: brew install chafa")

    return (f"[multimedia] Couldn't fetch a poster for {title} "
            f"(network issue or unknown title).")


def answer_crowdsource(movie, relation):
    if not movie or not relation:
        return ("Crowdsource lookup needs a movie + property. "
                "Try: 'is it true that the director of Inception is Christopher Nolan?'")
    facts = FACTS.get(movie, {})
    truth = facts.get(relation)
    # Look for a matching crowd record
    for rec in CROWD:
        m, r, _ = rec["claim"]
        if m == movie and r == relation:
            return (f"Crowd verdict on '{title_case(movie)} → {relation}':\n"
                    f"  Support: {rec['support']}  Reject: {rec['reject']}  "
                    f"Inter-rater agreement: {rec['agreement']:.2f}\n"
                    f"  KG ground truth: {truth or 'unknown'}")
    return (f"No crowd records for that exact claim. KG says: "
            f"{relation} of {title_case(movie)} is {truth or 'unknown'}.")


def process_question(question):
    q = question.lower().strip().rstrip("?.!")
    movie = find_movie(q)
    relation = find_relation(q)

    # Routing — same priority as TobyKh_Bot.process_question
    if any(t in q for t in MULTIMEDIA_TRIGGERS):
        return answer_multimedia(movie)
    if any(t in q for t in CROWD_TRIGGERS) and movie and relation:
        return answer_crowdsource(movie, relation)
    if any(t in q for t in RECOMMENDATION_TRIGGERS):
        return answer_recommendation(movie or "")
    if movie and relation:
        return answer_factual(movie, relation)
    if movie:
        return (f"I know about {title_case(movie)} but I'm not sure what you're asking. "
                f"Try: 'who directed {title_case(movie)}?'")
    return ("Sorry, I couldn't identify the movie. Try asking about: "
            + ", ".join(title_case(m) for m in list(FACTS.keys())[:5]) + ".")


# --- REPL -------------------------------------------------------------------
def repl():
    # Real data from the repo, if present — proves the demo is wired to actual files
    movie_map = load_json_quiet("movie_imdb_mapping.json")
    if movie_map:
        print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Loaded "
              f"{len(movie_map)} IMDb movie mappings from movie_imdb_mapping.json\n")

    while True:
        try:
            question = input(f"{Fore.MAGENTA}you ▸ {Style.RESET_ALL}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", ":q"}:
            print(f"{Fore.CYAN}bye!{Style.RESET_ALL}")
            break

        print(f"{Fore.YELLOW}[PROCESSING]{Style.RESET_ALL} parsing entities & relations...")
        time.sleep(0.4)
        response = process_question(question)
        print(f"{Fore.GREEN}[ANSWER]{Style.RESET_ALL}")
        slow_print(f"  {response}\n", delay=0.008)


def main():
    banner()
    fake_init()
    repl()


if __name__ == "__main__":
    main()
