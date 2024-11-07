import rdflib

class KnowledgeGraphQuery:
    def __init__(self, graph_path):
        # Load the knowledge graph
        self.graph = rdflib.Graph()
        self.graph.parse(graph_path, format="turtle")
        print(f"Loaded graph with {len(self.graph)} triples.")

    def get_movies_by_genre(self, genre):
        """
        SPARQL query to retrieve movies that belong to a specific genre.
        """
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
        results = self.graph.query(query)
        movies = [str(row.movieLabel) for row in results]
        return movies

    def get_movies_by_era(self, era):
        """
        SPARQL query to retrieve movies released in a specific era (e.g., 1980s).
        """
        start_year, end_year = self.parse_era(era)
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
        results = self.graph.query(query)
        movies = [str(row.movieLabel) for row in results]
        return movies

    def get_movies_by_genre_and_era(self, genre, era):
        """
        SPARQL query to retrieve movies with a specific genre and release year range.
        """
        start_year, end_year = self.parse_era(era)
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
        results = self.graph.query(query)
        movies = [str(row.movieLabel) for row in results]
        return movies

    def parse_era(self, era):
        """
        Helper function to convert an era (e.g., "1980s") into a start and end year.
        """
        if era.endswith("s"):
            start_year = int(era[:-1])
            end_year = start_year + 9
            return start_year, end_year
        return int(era), int(era)  # Single year if no "s" suffix

    def list_genres(self):
        """
        SPARQL query to retrieve all unique genres in the knowledge graph.
        """
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
        results = self.graph.query(query)
        genres = [str(row.genreLabel) for row in results]
        return genres

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