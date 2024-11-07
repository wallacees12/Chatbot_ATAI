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

DEFAULT_HOST_URL = 'https://speakeasy.ifi.uzh.ch'
listen_freq = 2

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
        doc = self.nlp(question)


        if self.is_reccomendation_request(question):
            info = self.extract_entites(question) 
            print(self.extract_entites(question))
            recommendations = self.get_recommendations(info['movie_titles'], num_recommendations=5,genre = info['genres'], era = info['eras'])
            response = self.format_reccomendations(recommendations)
            return response
        entities = [
            (ent.text, ent.label_) for ent in doc.ents
            if ent.label_ in ['PERSON', 'WORK_OF_ART', 'ORG', 'GPE', 'EVENT', 'PRODUCT']
        ]
        relation_label = self.extract_relation_spacy(doc)

        print(f"Extracted entities: {entities}")
        print(f"Extracted relation: {relation_label}")

        if entities and relation_label:
            entity_label = entities[0][0]
            print(f"Entity label: {entity_label}")
            entity_uri = self.match_entity(entity_label)
            print(f"Matched entity URI: {entity_uri}")
            relation_uri = self.match_relation_label(relation_label)
            print(f"Matched relation URI: {relation_uri}")

            if entity_uri and relation_uri:
                sparql_query = self.construct_sparql_query(entity_uri, relation_uri,relation_label)
                factual_answers = self.execute_sparql_query(sparql_query,relation_label)

                if not factual_answers:
                    embedding_answers = self.predict_with_embeddings(entity_uri, relation_uri)
                else:
                    embedding_answers = None

                response = self.get_response(entity_label, relation_label, factual_answers, embedding_answers)
            else:
                response = f"Sorry, I couldn't find information about '{entity_label}' or understand the relation '{relation_label}'."
        else:
            if entities:
                entity_label = entities[0][0]
                response = f"Sorry, I managed to find the entity {entity_label}, but was unable to parse your question, maybe try rephrasing?"
            if relation_label:
                response = f"Sorry, I understood your question about {relation_label} but didn't find the film, is it spelt correctly including capitalisation?"
            else:
                response =f"Sorry, I didn't quite get that, can you rephrase your question please."
        return response

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

    def extract_entites(self, question):

        doc = self.nlp(question)

        movie_titles = [ent.text for ent in doc.ents if ent.label_ == "WORK_OF_ART"]
        genres = []
        eras = []

        # Simple keyword matching for genres and eras
        with open("genres.txt", 'r') as file:
            genre_keywords = [line.strip() for line in file.readlines()]
        
        era_keywords = ["1970s", "1980s", "1990s","2000s","2010s","2020s","1970","1980","1990","2000","2010","2020","1950","1950s",
        "1950s","1940","1940s","1930","1930s","1920s","1920","1960","1960s" "classic", "old", "retro"]

        for token in doc:
            if token.text.lower() in genre_keywords:
                genres.append(token.text.lower())
            if token.text.lower() in era_keywords:
                eras.append(token.text.lower())

        return {
            "movie_titles": movie_titles,
            "genres": genres,
            "eras": eras
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

    def get_recommendations(self, movie_titles, num_recommendations=5, genre=None, era=None):
        all_recommendations = []

        # Loop through each movie title provided by the user
        if movie_titles:
            for movie_title in movie_titles:
                movie_uri = self.match_entity(movie_title)
                if movie_uri is None:
                    continue  # Skip if the movie title is not recognized

                movie_id = self.ent2id.get(movie_uri)
                if movie_id is None:
                    continue  # Skip if no embedding is available for this movie

                # Calculate similarity between the target movie embedding and all other movie embeddings
                target_embedding = self.entity_emb[movie_id].reshape(1, -1)
                similarities = cosine_similarity(target_embedding, self.entity_emb).flatten()

                # Get the indices of the most similar movies, excluding the target movie itself
                similar_indices = np.argsort(-similarities)[1:]

                # Filter recommendations by genre and era, if specified
                for idx in similar_indices:
                    similar_movie_uri = self.id2ent[idx]
                    similar_movie_title = self.ent2lbl.get(similar_movie_uri)

                    # Skip if no title is found
                    if not similar_movie_title:
                        continue
                    
                    # Optional: Check genre and era filters
                    if genre and genre not in self.get_movie_genre(similar_movie_uri):
                        continue
                    if era and era not in str(self.get_movie_year(similar_movie_uri)):
                        continue
                    
                    all_recommendations.append(similar_movie_title)
                    
                    # Stop if we reach a certain number of recommendations for each movie
                    if len(all_recommendations) >= num_recommendations * len(movie_titles) * 4:
                        break
        elif genre and era:
            # Get recommendations based on both genre and era
            all_recommendations = self.get_movies_by_genre_and_era(genre, era)
            if all_recommendations == []:
                print(f"No results found for '{genre}' in era '{era}'. Expanding to include related subgenres.")
                all_recommendations = self.get_movies_by_broadgenre_era(genre,era)
        elif genre:
            # Get recommendations based on genre only
            print(f"Looking for films with genre {genre}")
            all_recommendations = self.get_movies_by_genre(genre)
            if all_recommendations == []:
                print(f"No results found for '{genre}'. Expanding to include related subgenres.")
                all_recommendations = self.get_movies_by_broad(genre)
        elif era:
            # Get recommendations based on era only
            all_recommendations = self.get_movies_by_era(era)

        # Rank and prioritize recommendations that appear multiple times
        ranked_recommendations = Counter(all_recommendations)
        sorted_recommendations = [title for title, _ in ranked_recommendations.most_common(num_recommendations)]
        final_recommendations = [movie for movie in sorted_recommendations if movie not in movie_titles]

        print(final_recommendations[:num_recommendations])
        return final_recommendations[:num_recommendations]

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
        return subgenres


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
        print(results)
        movies = [str(row.movieLabel) for row in results]
        return movies

    def get_movies_by_era(self, eras):
        """
        SPARQL query to retrieve movies released in any of the specified eras (e.g., ["1980s", "1990"]).
        """
        # Parse the list of eras to get a list of (start_year, end_year) tuples
        parsed_eras = self.parse_era(eras)
        
        # Build the FILTER clause to match any of the specified eras
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
        
        # Execute the query
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
        results = self.graph.query(query)
        movies = [str(row.movieLabel) for row in results]
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
    def parse_era(self, eras):
        """
        Helper function to convert a list of eras (e.g., ["1980s", "1990s", "1985"]) 
        into a list of tuples with start and end years.
        """
        parsed_eras = []
        
        for era in eras:
            # Check if the era is a decade (e.g., "1980s")
            if era.endswith("s"):
                start_year = int(era[:-1])
                end_year = start_year + 9
                parsed_eras.append((start_year, end_year))
            else:
                # Treat as a single year if no "s" suffix
                year = int(era)
                parsed_eras.append((year, year))
        
        return parsed_eras

    @staticmethod
    def get_time():
        return time.strftime("%H:%M:%S, %d-%m-%Y", time.localtime())


if __name__ == '__main__':
    agent = Agent("timid-spirit", "B7uzR8A5")
    agent.listen()
