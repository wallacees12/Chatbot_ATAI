import rdflib
from functools import lru_cache

class KnowledgeGraphQuery:
    def __init__(self, graph_path):
        # Load the knowledge graph
        self.graph = rdflib.Graph()
        try:
            self.graph.parse(graph_path, format="turtle")
            print(f"Loaded graph with {len(self.graph)} triples.")
        except Exception as e:
            print(f"[ERROR] Failed to load graph: {e}")

    def get_movies_by_genre(self, genre):
        """Retrieve movies in a specific genre."""
        if not genre:
            print("[ERROR] Genre cannot be empty.")
            return []
        query = f"""
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?movieLabel WHERE {{
            ?movie wdt:P136 ?genre .
            ?genre rdfs:label "{genre}"@en .
            ?movie rdfs:label ?movieLabel .
            FILTER (lang(?movieLabel) = "en")
        }}
        LIMIT 50
        """
        try:
            results = self.graph.query(query)
            return [str(row.movieLabel) for row in results]
        except Exception as e:
            print(f"[ERROR] Failed to retrieve movies by genre '{genre}': {e}")
            return []

    def get_movies_by_era(self, era):
        """Retrieve movies released in a specific era."""
        start_year, end_year = self.parse_era(era)
        if not start_year or not end_year:
            print(f"[ERROR] Invalid era: {era}")
            return []
        query = f"""
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        
        SELECT ?movieLabel WHERE {{
            ?movie wdt:P577 ?date .
            FILTER (YEAR(?date) >= {start_year} && YEAR(?date) <= {end_year}) .
            ?movie rdfs:label ?movieLabel .
            FILTER (lang(?movieLabel) = "en")
        }}
        LIMIT 50
        """
        try:
            results = self.graph.query(query)
            return [str(row.movieLabel) for row in results]
        except Exception as e:
            print(f"[ERROR] Failed to retrieve movies for era '{era}': {e}")
            return []

    def get_movies_by_genre_and_era(self, genre, era):
        """Retrieve movies by genre and era."""
        start_year, end_year = self.parse_era(era)
        if not start_year or not end_year:
            print(f"[ERROR] Invalid era: {era}")
            return []
        if not genre:
            print("[ERROR] Genre cannot be empty.")
            return []
        query = f"""
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?movieLabel WHERE {{
            ?movie wdt:P136 ?genre ;
                   wdt:P577 ?date .
            ?genre rdfs:label "{genre}"@en .
            FILTER (YEAR(?date) >= {start_year} && YEAR(?date) <= {end_year}) .
            ?movie rdfs:label ?movieLabel .
            FILTER (lang(?movieLabel) = "en")
        }}
        LIMIT 50
        """
        try:
            results = self.graph.query(query)
            return [str(row.movieLabel) for row in results]
        except Exception as e:
            print(f"[ERROR] Failed to retrieve movies by genre '{genre}' and era '{era}': {e}")
            return []

    def parse_era(self, era):
        """Convert an era into a start and end year."""
        try:
            if era.endswith("s"):
                start_year = int(era[:-1])
                end_year = start_year + 9
                return start_year, end_year
            return int(era), int(era)
        except ValueError:
            print(f"[ERROR] Invalid era format: {era}")
            return None, None

    @lru_cache(maxsize=128)
    def list_genres(self):
        """Retrieve all unique genres in the knowledge graph."""
        query = """
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT DISTINCT ?genreLabel WHERE {
            ?movie wdt:P136 ?genre .
            ?genre rdfs:label ?genreLabel .
            FILTER (lang(?genreLabel) = "en")
        }
        ORDER BY ?genreLabel
        """
        try:
            results = self.graph.query(query)
            return [str(row.genreLabel) for row in results]
        except Exception as e:
            print(f"[ERROR] Failed to list genres: {e}")
            return []


# Initialize the KnowledgeGraphQuery with the path to the .nt file
kg_query = KnowledgeGraphQuery("14_graph.nt")

# Test querying by genre
genre = "horror"
print(f"Movies in the genre '{genre}':")
movies_by_genre = kg_query.get_movies_by_genre(genre)
print(movies_by_genre)

# Test querying by era
era = "1980s"
print(f"\nMovies from the era '{era}':")
movies_by_era = kg_query.get_movies_by_era(era)
print(movies_by_era)

# Test querying by genre and era
print(f"\nMovies in the genre '{genre}' from the era '{era}':")
movies_by_genre_and_era = kg_query.get_movies_by_genre_and_era(genre, era)
print(movies_by_genre_and_era)

print("Genres available in the knowledge graph:")
genres = kg_query.list_genres()
for genre in genres:
    print(genre)