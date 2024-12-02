from speakeasypy import Speakeasy, Chatroom
from typing import List
import time
import timeit
import rdflib
import os
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import spacy
import difflib
from spacy.pipeline import EntityRuler
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from colorama import Fore, Style  # For colored console output
import json
import torch
from torchvision.io import read_image
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
import csv


DEFAULT_HOST_URL = 'https://speakeasy.ifi.uzh.ch'
listen_freq = 2
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
device = torch.device("cpu")

class Agent:
    def __init__(self, username, password):
        # Start timing the initialization
        init_start = timeit.default_timer()

        # Print header for initialization
        print("="*40)
        print("          INITIALIZING AGENT")
        print("="*40)

        # Set username and initialize the Speakeasy connection
        username_start = timeit.default_timer()
        self.username = username
        self.speakeasy = Speakeasy(host=DEFAULT_HOST_URL, username=username, password=password)
        self.speakeasy.login()
        username_time = timeit.default_timer() - username_start
        print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Speakeasy connection and login time: {username_time:.4f} seconds")

        # Load and parse the RDF graph
        graph_start = timeit.default_timer()
        self.graph = rdflib.Graph()
        self.graph.parse('14_graph.nt', format='turtle')
        graph_time = timeit.default_timer() - graph_start
        print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Graph loading and parsing time: {graph_time:.4f} seconds")
        
        self.crowd_data = self.load_crowd_data("crowd_data.tsv")
        
        # Build dictionaries for labels, entities, and relations
        label_to_entity_start = timeit.default_timer()
        self.label_to_entity = self.build_label_to_entity_dict()
        self.ent2lbl = {ent: str(lbl) for ent, lbl in self.graph.subject_objects(rdflib.RDFS.label)}
        self.relation_to_uri = self.build_relation_to_uri_dict()
        label_to_entity_time = timeit.default_timer() - label_to_entity_start
        print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Label and relation dictionary build time: {label_to_entity_time:.4f} seconds")

        # Load embeddings
        embeddings_start = timeit.default_timer()
        self.entity_emb, self.relation_emb, self.ent2id, self.id2ent, self.rel2id, self.id2rel = self.load_embeddings()
        embeddings_time = timeit.default_timer() - embeddings_start
        print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Embeddings loading time: {embeddings_time:.4f} seconds")

        # Load spaCy model and add movie title patterns
        nlp_start = timeit.default_timer()
        self.nlp = spacy.load('en_core_web_md')
        self.nlp = self.add_movie_title_patterns(self.nlp)
        nlp_time = timeit.default_timer() - nlp_start
        print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} spaCy model loading and movie title patterns setup time: {nlp_time:.4f} seconds")
        
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.eval()
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.preprocess = ResNet18_Weights.DEFAULT.transforms()
        
        # Print total initialization time
        init_time = timeit.default_timer() - init_start
        print("="*40)
        print(f"Total Initialization Time: {init_time:.4f} seconds")
        print("="*40)

        # ASCII banner to indicate the bot is live
        print("\n\n")
        print("╔═══════════════════════════════════════════════════════╗")
        print("║                    BOT IS NOW LIVE!                   ║")
        print("╚═══════════════════════════════════════════════════════╝")
        print(''' 
              ---                                    
           -        --                               
       --( /     \\ )XXXXXXXXXXXXX                   
   --XXX(   O   O  )XXXXXXXXXXXXXXX-              
  /XXX(       U     )        XXXXXXX\\               
/XXXXX(              )--   XXXXXXXXXXX\\             
/XXXXX/ (      O     )   XXXXXX   \\XXXXX\\
XXXXX/   /            XXXXXX   \\   \\XXXXX----        
XXXXXX  /          XXXXXX         \\  ----  -         
XXX  /          XXXXXX      \\           ---        
  --  /      /\\  XXXXXX            /     ---=         
   /    XXXXXX              '--- XXXXXX         
--\\/XXX\\ XXXXXX                      /XXXXX         
   \\XXXXXXXXX                        /XXXXX/
    \\XXXXXX                         /XXXXX/         
      \\XXXXX--  /                -- XXXX/       
       --XXXXXXX---------------  XXXXX--         
          \\XXXXXXXXXXXXXXXXXXXXXXXX-            
            --XXXXXXXXXXXXXXXXXX-
        
Hello! You can ask me factual questions about movies.
        ''')


    def is_reccomendation_request(self,question):
        doc = self.nlp(question)
        question = question.lower()

        # Define possible recommendation patterns based on dependency structure
        recommendation_patterns = [
            ("recommend", ["movies", "films", "similar", "like"]),
            ("suggest", ["movies", "films", "similar", "like"]),
            ("like", ["movies", "films"]),
            ("similar", ["movies", "films"])
        ]
        recommendation_phrases = [
        "can you recommend", "recommend", "recommend something", "recommend me",
        "can you suggest", "suggest something", "suggest movies", "suggest films",
        "any suggestions", "any recommendations", "I want a recommendation",
        "movies like", "films like", "similar to", "something similar to",
        "something like", "anything like", "anything similar to",
        "movies that are similar to", "films that are similar to",
        "find movies similar to", "find films like", "find something like",
        "looking for movies like", "looking for films like", "find movies similar to",
        "show me movies like", "show me films like", "give me movies similar to",
        "any movies similar to", "any films similar to",
        "recommend based on", "movies along the lines of",
        "I like movies like", "I love movies like", "films that remind me of",
        "suggest films that are like", "suggest movies that are like",
        "do you know movies like", "do you know films like",
        "find a similar movie", "find a similar film",
        "suggest another movie like", "suggest another film like",
        "recommend movies if I liked", "recommend films if I enjoyed",
        "anything in that style", "anything else like that",
        "can you find movies like", "can you find films like",
        "if you liked", "if you enjoy", "if you love", "I enjoy", "I enjoyed",
        "given that I like", "I like", "I love", "I enjoy",
        "given that I like movies like", "since I like",
        "recommend based on my interest in", "given my love for",
        "based on my taste for", "for someone who loves", "for a fan of",
        "give I like", "given my taste", "given my penchant for",
        "suppose I like",
        
        # Specific examples
        "Given that I like The Lion King, Pocahontas, and The Beauty and the Beast, can you recommend some movies?",
        "Recommend movies like Nightmare on Elm Street, Friday the 13th, and Halloween.",
        "I enjoy Disney movies like Aladdin and The Little Mermaid, any recommendations?",
        "I like horror movies from the 1980s, can you suggest similar films?",
        "Suggest movies similar to animated Disney classics like Snow White and Cinderella.",
        "Can you recommend some horror classics from the 1970s or 1980s?",
        "I loved watching classic animated movies, what should I watch next?",
        "I'm a fan of slasher films like Halloween and Scream, any other movies in that genre?",
        "For someone who loves animated movies, what do you recommend?",
        "What movies would you recommend to a Disney animation fan?",
        "Are there similar movies to The Lion King or other Disney classics?"
    ]

        if any(phrase in question for phrase in recommendation_phrases):
            return True

        for token in doc:
            for verb, keywords in recommendation_patterns:
                if token.lemma_ == verb:
                    # Check if any keyword appears in the sentence with the verb
                    for child in token.children:
                        if child.lemma_ in keywords:
                            return True
        return False

    def add_movie_title_patterns(self, nlp):

        ruler = nlp.add_pipe('entity_ruler', before='ner')

        movie_patterns = []
        with open('top_10000_movies_by_votes.txt', 'r') as file:
            for line in file:
                title = line.strip()
                # Create a pattern for each title to be recognized as a WORK_OF_ART entity
                movie_patterns.append({"label": "WORK_OF_ART", "pattern": title})
        
        # Add all movie patterns to the EntityRuler
        ruler.add_patterns(movie_patterns)
        patterns = [
            {
                "label": "WORK_OF_ART",
                "pattern": "Star Wars: Episode VI - Return of the Jedi"
            },
            {
                "label": "WORK_OF_ART",
                "pattern": "The Godfather"
            },
            {
                "label": "WORK_OF_ART",
                "pattern": "Apocalypse Now"
            },
            {
                "label": "WORK_OF_ART",
                "pattern": "The Masked Gang: Cyprus"
            },
            {
                "label": "WORK_OF_ART",
                "pattern": "Star Wars"
            },
            {
                "label": "WORK_OF_ART",
                "pattern": [{"LOWER": "star"}, {"LOWER": "wars"}, {"LOWER": "episode"}, {"IS_DIGIT": True}]
            },
            {
                "label": "WORK_OF_ART",
                "pattern": [{"LOWER": "episode"}, {"IS_DIGIT": True}, {"TEXT": "-", "OP": "?"}, {"LOWER": "return"}, {"LOWER": "of"}, {"LOWER": "the"}, {"LOWER": "jedi"}]
            },
            {
                "label": "WORK_OF_ART",
                "pattern": [{"TEXT": "The"}, {"TEXT": "Godfather"}]
            },
            {
                "label": "WORK_OF_ART",
                "pattern": [{"TEXT": "Apocalypse"}, {"TEXT": "Now"}]
            },
            # Add more patterns as needed
        ]
        ruler.add_patterns(patterns)
    
        return nlp


    def build_label_to_entity_dict(self):
        label_to_entity = {}
        for entity, label in self.graph.subject_objects(rdflib.RDFS.label):
            label_str = str(label).lower()
            label_to_entity[label_str] = entity
        return label_to_entity

    def build_relation_to_uri_dict(self):
        with open('relation_mappings.json', 'r', encoding='utf-8') as f:
            relations = json.load(f)
        for key in relations:
            relations[key] = rdflib.URIRef(relations[key])
        return relations

    def load_embeddings(self):
        entity_emb = np.load('entity_embeds.npy')
        relation_emb = np.load('relation_embeds.npy')

        ent2id = {}
        id2ent = {}
        with open('entity_ids.del', 'r') as f:
            for line in f:
                idx, entity = line.strip().split('\t')
                idx = int(idx)
                ent2id[rdflib.term.URIRef(entity)] = idx
                id2ent[idx] = rdflib.term.URIRef(entity)

        rel2id = {}
        id2rel = {}
        with open('relation_ids.del', 'r') as f:
            for line in f:
                idx, relation = line.strip().split('\t')
                idx = int(idx)
                rel2id[rdflib.term.URIRef(relation)] = idx
                id2rel[idx] = rdflib.term.URIRef(relation)

        return entity_emb, relation_emb, ent2id, id2ent, rel2id, id2rel

    def listen(self):
        while True:
            rooms: List[Chatroom] = self.speakeasy.get_rooms(active=True)
            for room in rooms:
                if not room.initiated:
                    room.post_messages("Hello! You can ask me factual questions about movies.")
                    room.initiated = True
                
                for message in room.get_messages(only_partner=True, only_new=True):
                    # Capture the current time for the incoming question
                    timestamp = self.get_time()

                    # [INCOMING QUESTION] Message
                    print(f"{Fore.RED}[INCOMING QUESTION]{Style.RESET_ALL}")
                    print(f"Chatroom ID: {room.room_id}")
                    print(f"Message #{message.ordinal}: '{message.message}'")
                    print(f"Timestamp: {timestamp}\n")
                    time.sleep(0.5)

                    # Process the question
                    question = message.message
                    print(f"{Fore.YELLOW}[PROCESSING BACKGROUND]{Style.RESET_ALL}")
                    response = self.process_question(question)
                    print("...processing complete.\n")

                    # [ANSWER RETURNED] Message
                    print(f"{Fore.GREEN}[ANSWER RETURNED]{Style.RESET_ALL}")
                    print(f"Response: {response}\n")

                    # Post response to the chatroom and mark message as processed
                    room.post_messages(response)
                    room.mark_as_processed(message)
                    
            time.sleep(listen_freq)


    def process_question(self, question):
        """
        Process a user's question and determine the appropriate response.
        """
        # Normalize the question for easier matching
        normalized_question = question.lower().strip().rstrip(".!?")

        # Use the NLP model to extract entities and additional information
        doc = self.nlp(question)
        info = self.extract_entities(question)
        print(info)

        # Extract relevant details
        movie_titles = info.get('movie_titles', [])
        cast_names = info.get('cast_names', [])
        entity_label = movie_titles[0] if movie_titles else (cast_names[0] if cast_names else None)

        # Handle multimedia-related questions
        multimedia_phrases = {
            "show me", "look like", "let me know what", "can you show", 
            "picture of", "image of", "appearance of", "frame of", "frames of", "scene of"
        }

        if any(phrase in normalized_question for phrase in multimedia_phrases):
            if "frames" in normalized_question:
                if entity_label:
                    return self.answer_movie_frames_question(entity_label)
                return "Sorry, I couldn't identify the movie you're asking about."
            if entity_label:
                return self.answer_multimedia_question(entity_label)
            return "Sorry, I couldn't find the person or movie you're referring to. Could you clarify?"

        # Handle recommendation questions
        if info['eras'] and not movie_titles:
            recommendations = self.get_movies_by_era(info['eras'])
            return self.format_reccomendations(recommendations)

        if self.is_reccomendation_request(question):
            print(f"[DEBUG] Extracted recommendation info: {info}")
            recommendations = self.get_recommendations(
                movie_titles, num_recommendations=5, genre=info['genres'], era=info['eras']
            )
            return self.format_reccomendations(recommendations)

        # Handle crowdsourcing-related questions
        if "box office" in normalized_question or "publication date" in normalized_question or "executive producer" in normalized_question:
            # Handle crowdsourcing questions related to movie data
            return self.answer_crowdsourcing_question(question)

        # Extract entities and relations
        entities = [
            (ent.text, ent.label_) for ent in doc.ents
            if ent.label_ in ['PERSON', 'WORK_OF_ART', 'ORG', 'GPE', 'EVENT', 'PRODUCT']
        ]
        relation_label = self.extract_relation_spacy(doc)

        print(f"Extracted entities: {entities}")
        print(f"Extracted relation: {relation_label}")

        # Handle factual or embedding questions
        if entities and relation_label:
            entity_label = entities[0][0]  # Take the first extracted entity
            print(f"Entity label: {entity_label}")
            entity_uri = self.match_entity(entity_label)
            print(f"Matched entity URI: {entity_uri}")
            relation_uri = self.match_relation_label(relation_label)
            print(f"Matched relation URI: {relation_uri}")

            if entity_uri and relation_uri:
                sparql_query = self.construct_sparql_query(entity_uri, relation_uri, relation_label)
                factual_answers = self.execute_sparql_query(sparql_query, relation_label)

                if not factual_answers:
                    embedding_answers = self.predict_with_embeddings(entity_uri, relation_uri)
                else:
                    embedding_answers = None

                return self.get_response(entity_label, relation_label, factual_answers, embedding_answers)

            return f"Sorry, I couldn't find information about '{entity_label}' or understand the relation '{relation_label}'."

        # Handle cases with partial information
        if entities:
            entity_label = entities[0][0]
            return f"Sorry, I managed to find the entity '{entity_label}', but was unable to parse your question. Could you try rephrasing?"

        if relation_label:
            return f"Sorry, I understood your question about '{relation_label}' but couldn't find the film. Is it spelled correctly?"

        # Fallback response for unrecognized questions
        return "Sorry, I didn't quite get that. Can you rephrase your question, please?"



    def extract_relation_spacy(self, doc):
        possible_relations = set(self.relation_to_uri.keys())
        with open('verb_to_relation.json', 'r', encoding='utf-8') as f:
            self.verb_to_relation = json.load(f)
        with open('question_word_to_relation.json', 'r', encoding='utf-8') as f:
            self.question_word_to_relation = json.load(f)

        interrogative = None
        for token in doc:
            if token.tag_ in ['WP', 'WRB']: 
                interrogative = token.text.lower()
                break
            
        for token in doc:
            if token.dep_ == 'attr' and token.lemma_.lower() in possible_relations:
                return token.lemma_.lower()

        if interrogative:
            mapping = self.question_word_to_relation.get(interrogative)
            if mapping:
                if isinstance(mapping, dict):
                    for token in doc:
                        lemma = token.lemma_.lower()
                        if lemma in mapping:
                            return mapping[lemma]
                else:
                    return mapping

        for token in doc:
            if token.pos_ == 'VERB':
                verb_lemma = token.lemma_.lower()
                if verb_lemma in self.verb_to_relation: 
                    return self.verb_to_relation[verb_lemma]

        for chunk in doc.noun_chunks:
            chunk_lemma = chunk.root.lemma_.lower()
            if chunk_lemma in possible_relations:
                return chunk_lemma

        return None
    def is_crowdsourcing_question(self, question):
        """
        Determine if the question relates to crowdsourcing topics like revenue, box office, etc.
        """
        crowdsourcing_keywords = ["box office", "votes", "crowdsourcing", "revenue", "publication date", "producer", "director"]
        # You could expand the list to include other keywords as needed.
        return any(keyword in question.lower() for keyword in crowdsourcing_keywords)

    def get_movie_genre(self, movie_uri):
        """
        Retrieve the genre(s) of a movie from the RDF graph using its URI.
        """
        query = f"""
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?genreLabel WHERE {{
            <{movie_uri}> wdt:P136 ?genre .
            ?genre rdfs:label ?genreLabel .
            FILTER (lang(?genreLabel) = "en")
        }}
        """
        try:
            results = self.graph.query(query)
            genres = [str(row['genreLabel']) for row in results]
            return genres
        except Exception as e:
            print(f"Error retrieving genres for {movie_uri}: {e}")
            return []

    def get_movie_year(self, movie_uri):
        """
        Retrieve the release year of a movie from the RDF graph using its URI.
        """
        query = f"""
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?year WHERE {{
            <{movie_uri}> wdt:P577 ?date .
            BIND(YEAR(?date) AS ?year)
        }}
        LIMIT 1
        """
        try:
            results = self.graph.query(query)
            for row in results:
                return int(row['year'])
        except Exception as e:
            print(f"Error retrieving year for {movie_uri}: {e}")
            return None

    def extract_entities(self, question):
        """
        Extracts entities like movie titles, cast names, genres, and eras from the question.
        """
        doc = self.nlp(question)

        # Extract movie titles from named entities
        movie_titles = [ent.text for ent in doc.ents if ent.label_ == "WORK_OF_ART"]

        # Extract cast names (actors/actresses)
        cast_names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

        # Load genre keywords
        with open("genres.txt", 'r') as file:
            genre_keywords = {line.strip().lower() for line in file}

        # Define era keywords
        era_keywords = {
            "1970s", "1980s", "1990s", "2000s", "2010s", "2020s",
            "classic", "old", "retro",
            "70s", "80s", "90s", "twentieth century"
        }

        # Extract genres
        genres = {token.text.lower() for token in doc if token.text.lower() in genre_keywords}

        # Extract explicit eras from era keywords
        detected_eras = {token.text.lower() for token in doc if token.text.lower() in era_keywords}

        # Infer eras based on movie titles' release years
        inferred_eras = [
            self.get_movie_year(self.match_entity(title))
            for title in movie_titles
            if self.match_entity(title) and self.get_movie_year(self.match_entity(title)) is not None
        ]
        inferred_eras = set(
            f"{int(year) // 10 * 10}s"
            for year in inferred_eras
            if year is not None
        )

        # Combine detected and inferred eras
        eras = detected_eras.union(inferred_eras)

        # Debugging logs
        if not (movie_titles or cast_names or genres or eras):
            print(f"[DEBUG] No entities extracted for question: {question}")

        print(f"Extracted entities: Movie Titles: {movie_titles}, Cast Names: {cast_names}, Genres: {genres}, Eras: {eras}")

        # Return all extracted entities
        return {
            "movie_titles": movie_titles,
            "cast_names": cast_names,
            "genres": list(genres),
            "eras": list(eras)
        }





    def match_entity(self, entity_text):
        labels = list(self.label_to_entity.keys())
        matches = difflib.get_close_matches(entity_text.lower(), labels, n=1, cutoff=0.5)
        if matches:
            matched_label = matches[0]
            return self.label_to_entity[matched_label]
        else:
            return None

    def match_relation_label(self, relation_label):
        return self.relation_to_uri.get(relation_label.lower())

    from collections import Counter

    def is_horror_movie(self, movie_uri):
        """
        Check if the movie belongs to the horror genre.
        """
        genres = self.get_movie_genre(movie_uri)
        return "Horror" in genres

    def get_recommendations(self, movie_titles, num_recommendations=5, genre=None, era=None):
        all_recommendations = []

        # If era is a list, extract the first value for processing
        if isinstance(era, list) and len(era) > 0:
            era = era[0]  # Use the first era for filtering
        elif isinstance(era, list):
            era = None  # Set to None if the list is empty

        if movie_titles:
            for movie_title in movie_titles:
                movie_uri = self.match_entity(movie_title)
                if not movie_uri:
                    print(f"Movie '{movie_title}' not recognized. Skipping.")
                    continue

                movie_id = self.ent2id.get(movie_uri)
                if movie_id is None:
                    print(f"Embeddings not found for '{movie_title}'. Skipping.")
                    continue

                target_embedding = self.entity_emb[movie_id].reshape(1, -1)
                similarities = cosine_similarity(target_embedding, self.entity_emb).flatten()

                similar_indices = np.argsort(-similarities)[1:]  # Exclude the movie itself
                for idx in similar_indices:
                    similar_movie_uri = self.id2ent[idx]
                    similar_movie_title = self.ent2lbl.get(similar_movie_uri)

                    # Skip if no title is found
                    if not similar_movie_title:
                        continue

                    # Retrieve genres for the similar movie
                    genres = self.get_movie_genre(similar_movie_uri) or []

                    # Convert genres to lowercase for case-insensitive comparison
                    genres_lower = [genre.lower() for genre in genres]

                    # Exclude movies classified as documentaries
                    if "documentary" in genres_lower:
                        print(f"Excluding {similar_movie_title} (documentary).")
                        continue

                    # Filter recommendations based on genre
                    if genre and genre.lower() not in genres_lower:
                        continue

                    # Filter recommendations based on era
                    if era and not self.is_within_era(similar_movie_uri, era):
                        continue

                    # Add valid recommendations
                    all_recommendations.append(similar_movie_title)

                    # Stop if we've gathered enough recommendations
                    if len(all_recommendations) >= num_recommendations * len(movie_titles):
                        break

        else:
            # Handle recommendations based only on genre and/or era
            if genre and era:
                all_recommendations = self.get_movies_by_genre_and_era(genre, era)
            elif genre:
                all_recommendations = self.get_movies_by_genre(genre)
            elif era:
                all_recommendations = self.get_movies_by_era(era)

        # Deduplicate and rank recommendations for diversity
        ranked_recommendations = Counter(all_recommendations)
        sorted_recommendations = [title for title, _ in ranked_recommendations.most_common(num_recommendations)]
        final_recommendations = [movie for movie in sorted_recommendations if movie not in movie_titles]

        print(f"Final Recommendations: {final_recommendations[:num_recommendations]}")
        return final_recommendations[:num_recommendations]


    def is_within_era(self, movie_uri, era):
        # Parse era into start and end year
        start_year, end_year = self.parse_era([era])[0]
        
        # Retrieve the movie's release year
        movie_year = self.get_movie_year(movie_uri)
        if not movie_year:
            return False

        # Check if the movie falls within the specified era
        return start_year <= movie_year <= end_year


    def construct_sparql_query(self, entity_uri, relation_uri, relation_label):
        
        # Query to get the director
        if relation_label == "director":  # Director relation
            query = f'''
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX wd: <http://www.wikidata.org/entity/>
            
            SELECT ?directorLabel WHERE {{
                <{entity_uri}> <{relation_uri}> ?director .
                ?director rdfs:label ?directorLabel .
                FILTER (lang(?directorLabel) = "en")
            }}
            LIMIT 5
            '''
        # Query to get the publication date
        elif relation_label == "publication date":  # Publication date relation
            query = f'''
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX wd: <http://www.wikidata.org/entity/>
            
            SELECT ?publicationDate WHERE {{
                <{entity_uri}> <{relation_uri}> ?publicationDate .
            }}
            LIMIT 5
            '''
            # Query to get the movie rating
        elif relation_label == "rating":
            query = f'''
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX wd: <http://www.wikidata.org/entity/>
            
            SELECT ?rating WHERE {{
                <{entity_uri}> <{relation_uri}> ?rating .
            }}
            LIMIT 1
            '''

        # Query to get the movie genre
        elif relation_label == "genre":
            query = f'''
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX wd: <http://www.wikidata.org/entity/>
            
            SELECT ?genreLabel WHERE {{
                <{entity_uri}> <{relation_uri}> ?genre .
                ?genre rdfs:label ?genreLabel .
                FILTER (lang(?genreLabel) = "en")
            }}
            LIMIT 3
            '''

        # Query to get the cast of the movie
        elif relation_label == "cast":
            query = f'''
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX wd: <http://www.wikidata.org/entity/>
            
            SELECT ?actorLabel WHERE {{
                <{entity_uri}> <{relation_uri}> ?actor .
                ?actor rdfs:label ?actorLabel .
                FILTER (lang(?actorLabel) = "en")
            }}
            LIMIT 10
            '''

        elif relation_label == "screenwriter":
            query = f'''
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX wd: <http://www.wikidata.org/entity/>

        SELECT ?screenwriterLabel WHERE {{
            <{entity_uri}> wdt:P58 ?screenwriter .  # P58 is the property for screenwriter
            ?screenwriter rdfs:label ?screenwriterLabel .
            FILTER (lang(?screenwriterLabel) = "en")
        }}
        LIMIT 5
        '''

        elif relation_label == "cast member":
            query = f'''
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX wd: <http://www.wikidata.org/entity/>

        SELECT ?castMemberLabel WHERE {{
            <{entity_uri}> <{relation_uri}> ?castMember .
            ?castMember rdfs:label ?castMemberLabel .
            FILTER (lang(?castMemberLabel) = "en")
        }}
        LIMIT 15
        '''

        else:
            query = f'''
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX wd: <http://www.wikidata.org/entity/>
            
            SELECT ?valueLabel WHERE {{
                <{entity_uri}> <{relation_uri}> ?value .
                ?value rdfs:label ?valueLabel .
                FILTER (lang(?valueLabel) = "en")
            }}
            LIMIT 5
            ''' 

        return query

    def execute_sparql_query(self, query, relation_label):
        results = self.graph.query(query)
        for row in results:
            print(row)  # Debug output

        # Handle different potential keys based on relation_label
        if relation_label == 'director':
            answers = [(str(row['directorLabel'])) for row in results]
        elif relation_label == 'publication date':
            answers = [(str(row['publicationDate'])) for row in results]
        elif relation_label == 'rating':
            answers = [(str(row['rating'])) for row in results]
        elif relation_label == 'genre':
            answers = [(str(row['genreLabel'])) for row in results]
        elif relation_label == 'cast':
            answers = [(str(row['actorLabel'])) for row in results]
        elif relation_label == "screenwriter":
            answers = [(str(row['screenwriterLabel'])) for row in results]
        elif relation_label == "cast member":
            answers = [(str(row['castMemberLabel'])) for row in results]
        else:
            answers = [(str(row['valueLabel'])) for row in results]
        return answers

    def predict_with_embeddings(self, entity_uri, relation_uri):
        head_id = self.ent2id.get(entity_uri)
        rel_id = self.rel2id.get(relation_uri)

        if head_id is None or rel_id is None:
            return None

        head_vector = self.entity_emb[head_id]
        rel_vector = self.relation_emb[rel_id]

        tail_vector = head_vector + rel_vector
        distances = pairwise_distances(tail_vector.reshape(1, -1), self.entity_emb).reshape(-1)
        closest_ids = distances.argsort()
        answers = []
        for idx in closest_ids[:3]:
            candidate_entity_uri = self.id2ent[idx]
            label = self.ent2lbl.get(candidate_entity_uri)
            if label:
                answers.append(label)
        return answers

    def get_response(self, entity_label, relation_label, factual_answers, embedding_answers):
        if factual_answers:
            answer_str = ', '.join(factual_answers)
            response = (
            f"The {relation_label} for {entity_label} is: {answer_str}.\n(Factual Answer)"
            ).encode('ascii', errors='replace').decode('ascii')
        elif embedding_answers:
            answer_str = (
            f"Our top pick is {embedding_answers[0]},\n "
            f"But it could also be {embedding_answers[1]}\n "
            f"Or perhaps {embedding_answers[2]}."
            )
            response = (
            f"We couldn't find the {relation_label} information for {entity_label} in our knowledge graph.\n\n"
            f"However, based on our embeddings:\n {answer_str}.\n (Embedding Answer)"
            ).encode('ascii', errors='replace').decode('ascii')
        else:
            response = f"Sorry, I couldn't find any information about the {relation_label} of {entity_label}."
        return response

    def format_reccomendations(self, recommendations):
        if not recommendations:
            return "I'm sorry, I couldn't find any movie recommendations based on your input."

        # Start building the recommendation message
        formatted_recommendations = "Here are some movies you might enjoy:\n\n"
        for idx, movie_title in enumerate(recommendations, start=1):
            formatted_recommendations += f"{idx}. {movie_title}\n"

        formatted_recommendations += "\nThese recommendations are based on the movies you like."
        formatted_recommendations += "\nHappy watching!"
        return formatted_recommendations


   
    def get_subgenres(self, genre_keywords):
        """
        Retrieves all subgenres that contain any of the specified genre keywords.
        The genre list is loaded from genres.txt, and subgenres are matched by checking
        if any of the keywords are present in each genre name.
        """
        # Load genres from genres.txt
        try:
            with open("genres.txt", "r") as file:
                genres = [line.strip() for line in file.readlines()]
        except FileNotFoundError:
            print("Error: 'genres.txt' file not found.")
            return []
        
        # Ensure genre_keywords is a list (even if a single keyword is passed)
        if isinstance(genre_keywords, str):
            genre_keywords = [genre_keywords]
        
        # Filter genres to find subgenres containing any keyword in genre_keywords
        subgenres = [
            genre for genre in genres 
            if any(keyword.lower() in genre.lower() for keyword in genre_keywords)
        ]
        print(f"Found all the subgenres {subgenres}")
        if not subgenres:
            print(f"No subgenres found for keywords: {genre_keywords}")
            
        return subgenres


    def get_movies_by_genre(self, genre):
        """
        SPARQL query to retrieve movies belonging to a specific genre.
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
        try:
            results = self.graph.query(query)
            movies = [str(row.movieLabel) for row in results]
        except Exception as e:
            print(f"Error retrieving movies by genre '{genre}': {e}")
            return []

        if not movies:
            print(f"No movies found for genre: {genre}")
            return []
        return movies


    def get_movies_by_era(self, eras):
        """
        SPARQL query to retrieve movies released within specified eras.
        """
        parsed_eras = self.parse_era(eras)
        if not parsed_eras:
            print(f"[DEBUG] No valid eras found in: {eras}")
            return []

        # Build the FILTER clause
        era_filters = " || ".join(
            f"(YEAR(?date) >= {start} && YEAR(?date) <= {end})" for start, end in parsed_eras
        )
        
        query = f"""
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?movieLabel WHERE {{
            ?movie wdt:P577 ?date .
            FILTER ({era_filters}) .
            ?movie rdfs:label ?movieLabel .
            FILTER (lang(?movieLabel) = "en")
        }}
        LIMIT 50
        """
        try:
            results = self.graph.query(query)
            movies = [str(row.movieLabel) for row in results]
        except Exception as e:
            print(f"Error retrieving movies for eras {eras}: {e}")
            return []

        if not movies:
            print(f"No movies found for eras: {eras}")
            return []
        return movies


   
    def get_movies_by_genre_and_era(self, genre, era):
        """
        SPARQL query to retrieve movies filtered by genre and release year range.
        """
        start_year, end_year = self.parse_era(era)
        query = f"""
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX wd: <http://www.wikidata.org/entity/>

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
            movies = [str(row.movieLabel) for row in results]
        except Exception as e:
            print(f"Error retrieving movies for genre '{genre}' and era '{era}': {e}")
            return []

        if not movies:
            print(f"No movies found for genre '{genre}' in era '{era}'")
            return []
        return movies


    def get_movies_by_broadgenre_era(self, broad_genre, era):
        """
        Expands the search for movies by genre and era to include all related subgenres if necessary.
        """
        # Get all subgenres related to the broad genre (e.g., all "horror" subgenres)
        subgenres = self.get_subgenres(broad_genre)

        # Parse the era to get the start and end year
        start_year, end_year = self.parse_era(era)

        # Build the SPARQL query to include all subgenres and filter by the era range
        subgenre_filters = " || ".join([f'?genreLabel = "{subgenre}"' for subgenre in subgenres])
        query = f"""
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?movieLabel WHERE {{
            ?movie wdt:P136 ?genre ;
                   wdt:P577 ?date .
            ?genre rdfs:label ?genreLabel .
            FILTER (lang(?genreLabel) = "en" && ({subgenre_filters})) .
            FILTER (YEAR(?date) >= {start_year} && YEAR(?date) <= {end_year}) .
            ?movie rdfs:label ?movieLabel .
        }}
        LIMIT 50
        """
        results = self.graph.query(query)
        movies = [str(row.movieLabel) for row in results]
        return movies

    def get_movies_by_broad(self, broad_genre):
        """
        Expands the search for movies by including all related subgenres when searching by genre alone.
        """
        # Get all subgenres related to the broad genre
        subgenres = self.get_subgenres(broad_genre)

        # Build the SPARQL query to include all subgenres
        subgenre_filters = " || ".join([f'?genreLabel = "{subgenre}"' for subgenre in subgenres])
        query = f"""
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?movieLabel WHERE {{
            ?movie wdt:P136 ?genre .
            ?genre rdfs:label ?genreLabel .
            FILTER (lang(?genreLabel) = "en" && ({subgenre_filters})) .
            ?movie rdfs:label ?movieLabel .
        }}
        LIMIT 50
        """
        results = self.graph.query(query)
        movies = [str(row.movieLabel) for row in results]
        return movies

    def preprocess_image(self, image_path):
        """
        Preprocess the input image for ResNet model.
        """
        original_img = read_image(image_path)
        preprocess_img = self.preprocess(original_img)  # Assumes self.preprocess() is defined elsewhere
        return original_img, preprocess_img.unsqueeze(0)

    def generate_image_response(self, image_path, entity_label):
        """
        Generate an HTML-compatible response with the image and ResNet visualization.
        """

        try:
            original_img, preprocess_img = self.preprocess_image(image_path)
            embedding = self.model(preprocess_img).detach().numpy().reshape(32, 64)

            # Create a figure to visualize the image and embedding
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            ax1.imshow(original_img.permute(1, 2, 0).numpy().astype('uint8'))
            ax1.set_title(entity_label)
            ax2.imshow(embedding, cmap='viridis')
            ax2.set_title("Embedding Visualization")

            # Convert the figure to a base64 image
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            encoded_image = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()

            # HTML response with embedded base64 image
            html_response = f"""
            <div>
                <h3>Image of {entity_label}</h3>
                <img src="data:image/png;base64,{encoded_image}" alt="Image of {entity_label}" />
            </div>
            """
            return html_response
        except Exception as e:
            return f"Sorry, an error occurred while processing the image: {str(e)}"

    def answer_multimedia_question(self, entity_label):
        """
        Retrieves a single image (including frames) for a cast member based on their name or IMDb ID,
        displays it, and includes the link in the chat response.
        """
        try:
            # Load datasets
            with open("actor_imdb_mapping.json", "r") as f:
                actor_mapping = json.load(f)
            with open("images.json", "r") as f:
                images_data = json.load(f)
        except FileNotFoundError as e:
            missing_file = str(e).split("'")[-2]
            return f"Sorry, the required dataset '{missing_file}' is missing. Please check if the file exists in the project directory."

        # Get IMDb ID for the actor
        imdb_id = actor_mapping.get(entity_label)
        if not imdb_id:
            return f"Sorry, I couldn't find IMDb information for {entity_label}."

        # Search for images and frames of the actor
        base_url = "https://files.ifi.uzh.ch/ddis/teaching/ATAI2024/dataset/movienet/images/"
        images = []
        frames = []

        for item in images_data:
            if imdb_id in item.get("cast", []):
                image_url = base_url + item["img"]
                if item.get("type") == "still_frame":
                    frames.append(image_url)
                else:
                    images.append(image_url)

        # Build the response message with URLs
        response_message = []

        # Show only one image if available
        if images:
            image_url = images[0]  # Get the first image
            response_message.append(f"Here is an image of {entity_label}:")
            response_message.append(f"Image: {image_url}")

            # Display the image directly in the chat
            try:
                response = requests.get(image_url)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content))

                # Display the image with matplotlib
                plt.imshow(img)
                plt.axis('off')
                plt.title(f"Image of {entity_label}")
                plt.show()
            except Exception as e:
                response_message.append(f"Sorry, I found an image for {entity_label}, but there was an issue displaying it.")
                print(f"[ERROR] Failed to display image: {e}")
        else:
            response_message.append(f"Sorry, I couldn't find any images for {entity_label}.")

        # Show only one frame if available
        if frames:
            frame_url = frames[0]  # Get the first frame
            response_message.append(f"\nHere is a frame of {entity_label}:")
            response_message.append(f"Frame: {frame_url}")

            # Display the frame directly in the chat
            try:
                response = requests.get(frame_url)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content))

                # Display the frame with matplotlib
                plt.imshow(img)
                plt.axis('off')
                plt.title(f"Frame of {entity_label}")
                plt.show()
            except Exception as e:
                response_message.append(f"Sorry, I found a frame for {entity_label}, but there was an issue displaying it.")
                print(f"[ERROR] Failed to display frame: {e}")
        else:
            response_message.append(f"Sorry, I couldn't find any frames for {entity_label}.")

        return "\n".join(response_message)

    def get_movie_frames(self, imdb_id):
        """
        Retrieves URLs of frames for a specific movie based on its IMDb ID.
        """
        base_url = "https://files.ifi.uzh.ch/ddis/teaching/ATAI2024/dataset/movienet/frames/"
        frames_url = base_url + imdb_id + "/"

        # Generate sample frame URLs (modify as needed to dynamically fetch all frames)
        return [
            f"{frames_url}shot_0000_img_0.jpg",
            f"{frames_url}shot_0000_img_1.jpg",
            f"{frames_url}shot_0001_img_0.jpg",
        ]

    def answer_movie_frames_question(self, movie_title):
        """
        Answers questions about movie frames.
        """
        imdb_id = self.get_imdb_id(movie_title)  # Implement or use a mapping function
        if not imdb_id:
            return f"Sorry, I couldn't find frames for {movie_title}."

        frames = self.get_movie_frames(imdb_id)
        if not frames:
            return f"Sorry, no frames are available for {movie_title}."

        # Only show one frame
        frame_url = frames[0]
        return f"Here is a frame from {movie_title}:\nFrame: {frame_url}"

    def get_imdb_id(self, movie_title):
        """
        Maps movie titles to IMDb IDs using the generated movie_imdb_mapping.json file.
        """
        try:
            with open("movie_imdb_mapping.json", "r") as f:
                movie_mapping = json.load(f)
            print(f"[INFO] Loaded movie mapping from movie_imdb_mapping.json")
        except (FileNotFoundError, json.JSONDecodeError):
            print("[ERROR] Movie mapping file not found or is invalid.")
            return None
        
        return movie_mapping.get(movie_title)

    def load_crowd_data(self, file_path):
        """
        Loads the crowd data from the TSV file and returns a list of dictionaries.
        """
        crowd_data = []
        try:
            with open(file_path, 'r') as file:
                reader = csv.DictReader(file, delimiter='\t')
                for row in reader:
                    crowd_data.append(row)
            print(f"[INFO] Loaded crowd data from {file_path}")
            return crowd_data
        except FileNotFoundError:
            print(f"Sorry, the file {file_path} was not found.")
            return []

        
    def get_movie_data(self, movie_title, column_name):
        """
        Retrieves specific data (e.g., box office, publication date, executive producer) for a movie.
        """
        for row in self.crowd_data:
            # Match the movie title with Input1ID, Input2ID, or Input3ID in the TSV
            if movie_title in (row['Input1ID'], row['Input2ID'], row['Input3ID']):
                return row.get(column_name, "Data not available.")
        print(f"[DEBUG] No data found for movie title: {movie_title}")  # Add debugging for data matching
        return "Sorry, no data found for {movie_title}."

   
    def get_answer_distribution(self, movie_title):
        """
        Retrieves the answer distribution and inter-rater agreement for a given movie title.
        """
        answer_counts = {"CORRECT": 0, "INCORRECT": 0}
        workers = []
        
        for row in self.crowd_data:
            # Match movie title with the subject (Input1ID), predicate (Input2ID), or object (Input3ID)
            if movie_title in (row['Input1ID'], row['Input2ID'], row['Input3ID']):
                answer_counts[row['AnswerLabel']] += 1
                workers.append(row['WorkerId'])
        
        if len(workers) > 1:
            # Calculate Fleiss' kappa (inter-rater agreement)
            kappa = self.calculate_fleiss_kappa(answer_counts)
        else:
            kappa = None  # Not enough raters to calculate Fleiss' kappa
        
        return answer_counts, kappa

    def calculate_fleiss_kappa(self, answer_counts):
        """
        Calculate Fleiss' kappa to measure inter-rater agreement.
        """
        total_ratings = sum(answer_counts.values())
        p_i = {answer: count / total_ratings for answer, count in answer_counts.items()}
        
        # Fleiss' kappa formula (simplified version for binary classification)
        P_e = sum([p * p for p in p_i.values()])
        P_o = sum([min(count, total_ratings - count) / total_ratings for count in answer_counts.values()])
        
        return (P_o - P_e) / (1 - P_e)
    
    
    def answer_crowdsourcing_question(self, question):
        """
        Answers crowdsourcing questions based on the question.
        """
        question = question.lower()

        if "box office" in question:
            movie_title = self.extract_movie_title(question)
            box_office = self.get_movie_data(movie_title, 'box_office')
            answer_distribution, kappa = self.get_answer_distribution(movie_title)
            answer = f"The box office of {movie_title} is {box_office}. [Crowd, inter-rater agreement {kappa if kappa else 'N/A'}, The answer distribution for this specific task was {answer_distribution['CORRECT']} support votes, {answer_distribution['INCORRECT']} reject votes]"
            return answer
        
        elif "publication date" in question:
            movie_title = self.extract_movie_title(question)
            publication_date = self.get_movie_data(movie_title, 'publication_date')
            answer_distribution, kappa = self.get_answer_distribution(movie_title)
            answer = f"The publication date of {movie_title} is {publication_date}. [Crowd, inter-rater agreement {kappa if kappa else 'N/A'}, The answer distribution for this specific task was {answer_distribution['CORRECT']} support votes, {answer_distribution['INCORRECT']} reject votes]"
            return answer
        
        elif "executive producer" in question:
            movie_title = self.extract_movie_title(question)
            executive_producer = self.get_movie_data(movie_title, 'executive_producer')
            answer_distribution, kappa = self.get_answer_distribution(movie_title)
            answer = f"The executive producer of {movie_title} is {executive_producer}. [Crowd, inter-rater agreement {kappa if kappa else 'N/A'}, The answer distribution for this specific task was {answer_distribution['CORRECT']} support votes, {answer_distribution['INCORRECT']} reject votes]"
            return answer

        return "Sorry, I couldn't understand the question."


    
    def extract_movie_title(self, question):
        """
        Extracts the movie title from the question by stripping unnecessary words.
        This is an improved version to handle various question patterns.
        """
        question = question.lower().strip().rstrip("?")  # Normalize the question
        # Define question patterns to identify the start of the movie title
        question_patterns = [
            "who is the executive producer of",
            "what is the box office of",
            "can you tell me the publication date of",
            "who directed",
            "who starred in"
        ]
        
        # Try to match the question with one of the patterns
        for pattern in question_patterns:
            if question.startswith(pattern):
                # Extract the movie title by removing the known pattern
                movie_title = question[len(pattern):].strip()
                print(f"[DEBUG] Extracted movie title: {movie_title}")  # Add debugging
                return movie_title
        
        # If no pattern matched, return an empty string
        return ""
    
    @staticmethod
    def get_time():
        return time.strftime("%H:%M:%S, %d-%m-%Y", time.localtime())


if __name__ == '__main__':
    agent = Agent("timid-spirit", "B7uzR8A5")
    agent.listen()
