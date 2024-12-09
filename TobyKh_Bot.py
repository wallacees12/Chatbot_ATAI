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
from colorama import Fore, Style
import json
import torch
from torchvision.io import read_image
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
from pprint import pprint
import csv
import pandas as pd

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
        # Process and aggregate crowd data
        self.crowd_aggregates = self.process_crowd_data(self.crowd_data)

        # Build dictionaries for labels, entities, and relations
        label_to_entity_start = timeit.default_timer()
        self.label_to_entity = self.build_label_to_entity_dict()
        self.load_hardcoded_labels("labels.json")
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

    def load_hardcoded_labels(self, filepath):
        """
        Loads a JSON file that contains 'entity_labels' and 'property_labels',
        and creates reverse mappings from label text to wikidata IDs.
        Also integrates these labels into label_to_entity for entity resolution.
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"[ERROR] {filepath} not found. No hardcoded mappings loaded.")
            self.label_to_wikidata = {}
            return

        entity_labels = data.get("entity_labels", {})
        property_labels = data.get("property_labels", {})

        self.label_to_wikidata = {}

        # For each entity label from the JSON:
        # 1. Add a reverse mapping (label to 'wd:Q####') in self.label_to_wikidata
        # 2. Also add to self.label_to_entity if it's not already there
        for qid, label in entity_labels.items():
            if label:
                normalized_label = label.lower()
                self.label_to_wikidata[normalized_label] = f"wd:{qid}"

                # Add to label_to_entity using a URIRef to represent the entity
                # 'wd:Q11621' -> 'http://www.wikidata.org/entity/Q11621'
                # Check if not already in label_to_entity to avoid overwriting 
                # graph-based label
                if normalized_label not in self.label_to_entity:
                    entity_uri = rdflib.URIRef(f"http://www.wikidata.org/entity/{qid}")
                    self.label_to_entity[normalized_label] = entity_uri

        # For properties, just do the label_to_wikidata mapping since they aren't 
        # part of the entity dictionary
        for pid, label in property_labels.items():
            if label:
                self.label_to_wikidata[label.lower()] = f"wdt:{pid}"

        print("[INFO] Hardcoded label mappings loaded and integrated into label_to_entity from labels.json")

    def process_crowd_data(self, crowd_data):
        """
        Aggregates crowd data by triple keys (Input1ID, Input2ID, Input3ID),
        counts correct/incorrect answers, and computes a simple inter-rater agreement.
        """
        if not crowd_data:
            print("[INFO] No crowd data loaded.")
            return {}

        df = pd.DataFrame(crowd_data)
        # Convert AnswerLabel to a uniform type (string)
        df['AnswerLabel'] = df['AnswerLabel'].astype(str)

        grouping_cols = ['Input1ID', 'Input2ID', 'Input3ID']

        def compute_inter_rater_agreement(correct_count, incorrect_count):
            total = correct_count + incorrect_count
            if total == 0:
                return 0.0
            return max(correct_count, incorrect_count) / total

        # Aggregate data
        agg = df.groupby(grouping_cols).apply(lambda g: pd.Series({
            'correct_count': (g['AnswerLabel'] == 'CORRECT').sum(),
            'incorrect_count': (g['AnswerLabel'] == 'INCORRECT').sum(),
            'crowd_answer_value': g['Input3ID'].iloc[0]  # Directly using Input3ID as the answer
        })).reset_index()

        agg['inter_rater_agreement'] = agg.apply(
            lambda row: compute_inter_rater_agreement(row['correct_count'], row['incorrect_count']), axis=1
        )
        agg['answer_distribution'] = agg.apply(
            lambda row: f"{row['correct_count']} support votes, {row['incorrect_count']} reject votes", axis=1
        )

        results_dict = {}
        for _, row in agg.iterrows():
            triple_key = (row['Input1ID'], row['Input2ID'], row['Input3ID'])
            results_dict[triple_key] = {
                'inter_rater_agreement': row['inter_rater_agreement'],
                'answer_distribution': row['answer_distribution'],
                'crowd_answer_value': row['crowd_answer_value']
            }
        return results_dict

    def is_reccomendation_request(self,question):
        doc = self.nlp(question)
        question = question.lower()

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
    ]

        if any(phrase in question for phrase in recommendation_phrases):
            return True

        for token in doc:
            for verb, keywords in recommendation_patterns:
                if token.lemma_ == verb:
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
                movie_patterns.append({"label": "WORK_OF_ART", "pattern": title})
        
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
                    
                    print(f"{Fore.RED}[INCOMING QUESTION]{Style.RESET_ALL}")
                    print(f"Chatroom ID: {room.room_id}")
                    print(f"Message #{message.ordinal}: '{message.message}'")
                    print(f"Timestamp: {self.get_time()}\n")
                    time.sleep(0.5)

                    question = message.message
                    print(f"{Fore.YELLOW}[PROCESSING BACKGROUND]{Style.RESET_ALL}")
                    response = self.process_question(question)
                    print("...processing complete.\n")

                    print(f"{Fore.GREEN}[ANSWER RETURNED]{Style.RESET_ALL}")
                    print(f"Response: {response}\n")

                    room.post_messages(response)
                    room.mark_as_processed(message)
                    
            time.sleep(listen_freq)

    def process_question(self, question):
        """
        Processes the user's question by first checking hardcoded mappings for entities and relations.
        If not found, falls back to normal NER and relation extraction.
        Then determines the type of request (multimedia, recommendation, factual) and responds accordingly.
        """
        # Normalize the question
        normalized_question = question.lower().strip().rstrip(".!?")


        # Step 1: Check Hardcoded Mappings First
        detected_mappings = []
        for label_text, wikidata_id in self.label_to_wikidata.items():
            if label_text in normalized_question:
                detected_mappings.append((label_text, wikidata_id))

        entity_uri = None
        relation_uri = None
        entity_label = None
        relation_label = None

        # Distinguish between entities and properties in detected mappings
        for label_text, wikidata_id in detected_mappings:
            if wikidata_id.startswith("wd:") and not entity_uri:
                entity_uri = wikidata_id
                entity_label = label_text  # Assuming label_text is the entity label
            elif wikidata_id.startswith("wdt:") and not relation_uri:
                relation_uri = wikidata_id
                relation_label = label_text  # Assuming label_text is the relation label

        # Debug print after hardcoded mapping
        print("[DEBUG] After hardcoded mapping:")
        print(f"Entity Label: {entity_label}, Entity URI: {entity_uri}")
        print(f"Relation Label: {relation_label}, Relation URI: {relation_uri}")

        # Step 2: Fallback to Normal Extraction if Needed
        if not entity_uri or not relation_uri:
            doc = self.nlp(question)
            info = self.extract_entities(question)

            if not entity_uri:
                entity_label = self.get_main_entity(info, doc)
                if entity_label:
                    entity_uri = self.match_entity(entity_label)

            if not relation_uri:
                relation_label = self.extract_relation_spacy(doc)
                if relation_label:
                    relation_uri = self.match_relation_label(relation_label)

            # Debug print after normal extraction
            print("[DEBUG] After normal extraction:")
            print(f"Entity Label: {entity_label}, Entity URI: {entity_uri}")
            print(f"Relation Label: {relation_label}, Relation URI: {relation_uri}")

        # Step 3: Validate Extraction Results
        if not entity_uri:
            return "Sorry, I couldn't identify the entity you're asking about."

        if not relation_uri and not entity_label:
            return "Sorry, I couldn't understand what property you're asking about."

        # Step 4: Retrieve Labels if Missing
        if not relation_label:
            relation_label = "this property"

        if not entity_label and entity_uri:
            uri_ref = rdflib.URIRef(entity_uri.replace("wd:", "http://www.wikidata.org/entity/"))
            entity_label = self.ent2lbl.get(uri_ref, "this entity")

        # Step 5: Determine the Type of Request
        multimedia_phrases = {
            "show me", "look like", "let me know what", "can you show", 
            "picture of", "image of", "appearance of", "frame of", "frames of", "scene of"
        }


        print(f"[DEBUG] Normalized Question: '{normalized_question}'")  # Added debug statement
        if any(phrase in normalized_question for phrase in multimedia_phrases):
            print("["*10 +"MULTIMEDIA"+"]"*10)
            print("*"*10)
            # Handle Multimedia Requests
            if "frames" in normalized_question and entity_label:
                return self.answer_movie_frames_question(entity_label)
            elif entity_label:
                return self.answer_multimedia_question(entity_label)
            else:
                return "Sorry, I couldn't find the person or movie you're referring to. Could you clarify?"

        if self.is_reccomendation_request(question):
            # Handle Recommendation Requestsa
            print("[DEBUG] Identified recommendation request.")
            recommendations = self.get_recommendations(
                info.get('movie_titles', []), 
                num_recommendations=5, 
                genre=info.get('genres', []), 
                era=info.get('eras', [])
            )
            return self.format_reccomendations(recommendations)

        # Step 6: Handle Factual Questions
        return self.answer_factual_question(entity_label, relation_label, entity_uri, relation_uri)

    def get_main_entity(self, info, doc):
        """
        Select a primary entity from the extracted information.
        Priority: movie_titles > cast_names > other recognized entities in doc.
        """
        movie_titles = info.get('movie_titles', [])
        cast_names = info.get('cast_names', [])

        # If no movie or cast, try generic entities from doc.ents
        if not movie_titles and not cast_names:
            generic_entities = [ent.text for ent in doc.ents if ent.label_ in ['PERSON','ORG','WORK_OF_ART','GPE','EVENT','PRODUCT']]
            if generic_entities:
                return generic_entities[0].strip()

        if movie_titles:
            return movie_titles[0].strip()
        if cast_names:
            return cast_names[0].strip()

        return None

    def answer_factual_question(self, entity_label, relation_label, entity_uri=None, relation_uri=None):
        """
        Attempts to answer a factual question using:
        1. Crowd data
        2. Knowledge Graph (KG)
        3. Embedding-based fallback
        """
        print("*"*60)
        print(f"Factual question about {entity_label} and relation {relation_label}")
        print("*"*60)

        # If entity_uri not provided, try to derive it from entity_label
        if not entity_uri:
            entity_uri = self.match_entity(entity_label)
            if not entity_uri:
                return f"Sorry, I couldn't find information about '{entity_label}'."

        # If relation_uri not provided, try to derive it from relation_label
        if not relation_uri:
            relation_uri = self.match_relation_label(relation_label)
            if not relation_uri:
                return f"Sorry, I couldn't find a known relation for '{relation_label}'."

        entity_uri = self.short_form_uri(entity_uri)
        relation_uri = self.short_form_uri(relation_uri)

        print("*"*60)
        print(f"Querying Crowdsource euri: {entity_uri} and ruri: {relation_uri}")
        print("*"*60)

        triple_key = self.get_triple_key_from_entity_and_relation(entity_uri, relation_uri)

        
        # Attempt crowd data
        crowd_response = self.answer_crowdsourcing_question(entity_label, relation_label, triple_key)

        # Attempt KG
        kg_answers = self.get_kg_answers(entity_uri, relation_uri, relation_label)

        # Attempt embeddings if KG fails and no crowd data

        if not kg_answers and not crowd_response:

            print("*"*60)
            print("Attempting embedding")
            print("*"*60)
            embedding_answers = self.predict_with_embeddings(self.lengthen_uri(entity_uri), self.lengthen_uri(relation_uri))
            if embedding_answers:
                return self.format_embedding_answer(entity_label, relation_label, embedding_answers)
            else:
                return f"Sorry, I couldn't find any information about the {relation_label} of {entity_label}."

        # Combine results
        final_response = ""
        if crowd_response:
            final_response += crowd_response
        if kg_answers:
            kg_answer_str = ', '.join(kg_answers)
            if final_response:
                final_response += f"According to the knowledge graph, the {relation_label} is: {kg_answer_str}."
            else:
                final_response = f"The {relation_label} of {entity_label} according to the knowledge graph is: {kg_answer_str}."

        if not final_response:
            # If somehow we got here with no crowd_response and no kg, fallback handled above.
            return f"Sorry, I couldn't find any information about the {relation_label} of {entity_label}."

        return final_response    

    def get_crowd_answer_value(self, triple_key):
        """
        Retrieves the most supported answer from crowd data for the given triple_key.

        Args:
            triple_key (tuple): The triple key (Input1ID, Input2ID, Input3ID).

        Returns:
            str or None: The most supported crowd answer or None if unavailable.
        """
        data = self.crowd_aggregates.get(triple_key)
        if not data:
            return None

        return data.get('crowd_answer_value')

    def answer_crowdsourcing_question(self, entity_label, relation_label, triple_key):
        """
        Given the entity_label, relation_label, and triple_key, returns the formatted crowd-sourced response.
        
        Args:
            entity_label (str): The label of the entity (e.g., movie title).
            relation_label (str): The label of the relation (e.g., "box office").
            triple_key (tuple): The triple key (Input1ID, Input2ID, Input3ID).
        
        Returns:
            str: The formatted response based on crowd data, or an apology if no data is found.
        """
        # Check if we have crowd data for this triple
        if triple_key not in self.crowd_aggregates:
            return ""
        
        data = self.crowd_aggregates[triple_key]
        irr = data['inter_rater_agreement']
        distribution = data['answer_distribution']
        
        # Retrieve the crowd answer value from the precomputed data
        crowd_answer_value = self.get_crowd_answer_value(triple_key)

        # Function to determine if a string is a QID
        def is_qid(value):
            if isinstance(value, str):
                if value.startswith('wd:'):
                    value = value[3:]
                return value.startswith('Q') and value[1:].isdigit()
            return False

        # If the answer is a QID, retrieve its label
        if is_qid(crowd_answer_value):
            # Construct the full URI for the QID
            qid_uri = f"http://www.wikidata.org/entity/{crowd_answer_value}"
            # Retrieve the label from ent2lbl
            crowd_answer_label = self.ent2lbl.get(rdflib.URIRef(qid_uri))
            print(crowd_answer_label)
            if crowd_answer_label:
                crowd_answer_value = crowd_answer_label
            else:
                # If label not found, retain the QID
                print(f"[WARNING] Label for QID {crowd_answer_value} not found.")
            
        if crowd_answer_value:
            response = (
                f"The {relation_label} of {entity_label} is {crowd_answer_value}.\n"
                f"[Crowd, inter-rater agreement {irr:.3f}, The answer distribution was {distribution}]\n\n"
        )
        else:
            # If no direct answer value is found, just report the distribution and IRR
            response = (
                f"[Crowd data available but no direct answer extracted for {relation_label} of {entity_label}. "
                f"Inter-rater agreement {irr:.3f}, distribution {distribution}]"
            )
        
        return response

    def get_kg_answers(self, entity_uri, relation_uri, relation_label):
        """
        Queries the knowledge graph for answers.
        """
        entity_uri = self.lengthen_uri(entity_uri)
        relation_uri = self.lengthen_uri(relation_uri)

        print("*"*60)
        print("Querying the knowledge graph")
        print("*"*60)
        sparql_query = self.construct_sparql_query(entity_uri, relation_uri, relation_label)
        kg_answers = self.execute_sparql_query(sparql_query, relation_label)
        print(kg_answers)
        return kg_answers

    def get_embedding_answers(self, entity_uri, relation_uri):
        """
        Uses embedding-based predictions as a fallback.
        """
        return self.predict_with_embeddings(entity_uri, relation_uri)

    def format_embedding_answer(self, entity_label, relation_label, embedding_answers):
        answer_str = (
            f"Our top pick is {embedding_answers[0]},\n "
            f"But it could also be {embedding_answers[1]}\n "
            f"Or perhaps {embedding_answers[2]}."
        )
        response = (
            f"We couldn't find the {relation_label} information for {entity_label} in our knowledge graph or crowd data.\n\n"
            f"However, based on our embeddings:\n {answer_str}.\n (Embedding Answer)"
        )
        return response

    def lengthen_uri(self, short_uri):
        """
        Converts a short URI (e.g., 'wd:Q457180') to its full URI (e.g., 'http://www.wikidata.org/entity/Q457180').

        Args:
            short_uri (str): The short form URI to be converted.

        Returns:
            str: The full URI corresponding to the short URI.

        Raises:
            ValueError: If the short_uri does not start with a recognized prefix.
        """
        # Define the mapping from short prefixes to full URI bases
        prefix_mapping = {
            'wd:': 'http://www.wikidata.org/entity/',
            'wdt:': 'http://www.wikidata.org/prop/direct/',
            'wds:': 'http://www.wikidata.org/entity/statement/',  # Example for other prefixes
            # Add more prefixes as needed
        }

        # Iterate through the prefix mapping to find a matching prefix
        for prefix, full_base in prefix_mapping.items():
            if short_uri.startswith(prefix):
                # Extract the identifier part after the prefix
                identifier = short_uri[len(prefix):]
                # Construct and return the full URI
                return f"{full_base}{identifier}"

        # If no matching prefix is found, raise an error
        raise ValueError(f"Unrecognized URI prefix in '{short_uri}'.")

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

    def extract_entities(self, question):
        doc = self.nlp(question)

        movie_titles = [ent.text for ent in doc.ents if ent.label_ == "WORK_OF_ART"]
        cast_names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

        with open("genres.txt", 'r') as file:
            genre_keywords = {line.strip().lower() for line in file}
        era_keywords = {
            "1970s", "1980s", "1990s", "2000s", "2010s", "2020s",
            "classic", "old", "retro",
            "70s", "80s", "90s", "twentieth century"
        }

        genres = {token.text.lower() for token in doc if token.text.lower() in genre_keywords}
        detected_eras = {token.text.lower() for token in doc if token.text.lower() in era_keywords}

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

        eras = detected_eras.union(inferred_eras)

        print(f"Extracted entities: Movie Titles: {movie_titles}, Cast Names: {cast_names}, Genres: {genres}, Eras: {eras}")

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

    def is_within_era(self, movie_uri, era):
        start_year, end_year = self.parse_era([era])[0]
        movie_year = self.get_movie_year(movie_uri)
        if not movie_year:
            return False
        return start_year <= movie_year <= end_year

    def get_triple_key_from_entity_and_relation(self, entity_uri, relation_uri):
        """
        attempts to find a triple key in self.crowd_aggregates that matches the given entity_uri and relation_uri.
        the triple key is of the form (input1id, input2id, input3id).
        
        assumptions:
        - entity_uri corresponds to input1id
        - relation_uri corresponds to input2id
        - input3id is unknown; we pick the first triple found.
        
        if no match is found, returns none.
        """
        
        for triple_key in self.crowd_aggregates.keys():
            # triple_key is something like (input1id, input2id, input3id)
            # check if the first two elements match our entity and relation uris
            if triple_key[0] == str(entity_uri) and triple_key[1] == str(relation_uri):
                print(f"*"*10)
                print("triple key in crowdsource")
                print("*"*10)
                return triple_key

        return None

    def construct_sparql_query(self, entity_uri, relation_uri, relation_label):
        if relation_label == "director":
            query = f'''
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            SELECT ?directorLabel WHERE {{
                <{entity_uri}> <{relation_uri}> ?director .
                ?director rdfs:label ?directorLabel .
                FILTER (lang(?directorLabel) = "en")
            }} LIMIT 5
            '''
        elif relation_label == "publication date":
            query = f'''
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            SELECT ?publicationDate WHERE {{
                <{entity_uri}> <{relation_uri}> ?publicationDate .
            }} LIMIT 5
            '''
        elif relation_label == "rating":
            query = f'''
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            SELECT ?rating WHERE {{
                <{entity_uri}> <{relation_uri}> ?rating .
            }} LIMIT 1
            '''
        elif relation_label == "genre":
            query = f'''
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?genreLabel WHERE {{
                <{entity_uri}> <{relation_uri}> ?genre .
                ?genre rdfs:label ?genreLabel .
                FILTER (lang(?genreLabel) = "en")
            }} LIMIT 3
            '''
        elif relation_label == "cast":
            query = f'''
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            SELECT ?actorLabel WHERE {{
                <{entity_uri}> <{relation_uri}> ?actor .
                ?actor rdfs:label ?actorLabel .
                FILTER (lang(?actorLabel) = "en")
            }} LIMIT 10
            '''
        elif relation_label == "screenwriter":
            query = f'''
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            SELECT ?screenwriterLabel WHERE {{
                <{entity_uri}> wdt:P58 ?screenwriter .
                ?screenwriter rdfs:label ?screenwriterLabel .
                FILTER (lang(?screenwriterLabel) = "en")
            }} LIMIT 5
            '''
        elif relation_label == "cast member":
            query = f'''
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            SELECT ?castMemberLabel WHERE {{
                <{entity_uri}> <{relation_uri}> ?castMember .
                ?castMember rdfs:label ?castMemberLabel .
                FILTER (lang(?castMemberLabel) = "en")
            }} LIMIT 15
            '''
        else:
            query = f'''
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            SELECT ?valueLabel WHERE {{
                <{entity_uri}> <{relation_uri}> ?value .
                ?value rdfs:label ?valueLabel .
                FILTER (lang(?valueLabel) = "en")
            }} LIMIT 5
            '''
        return query

    def execute_sparql_query(self, query, relation_label):
        results = self.graph.query(query)
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
        print("*"*60)
        print(f"Predicting euri: {entity_uri} -> ruri: {relation_uri} with embeddings")
        print("*"*60)

        entity_uri_ref = rdflib.URIRef(entity_uri)
        relation_uri_ref = rdflib.URIRef(relation_uri)

        print(f"[DEBUG] Converted Entity URI to rdflib.URIRef: {entity_uri_ref}")
        print(f"[DEBUG] Converted Relation URI to rdflib.URIRef: {relation_uri_ref}")


        head_id = self.ent2id.get(entity_uri_ref)
        rel_id = self.rel2id.get(relation_uri_ref)

        print(f"Head_id {head_id} and Tail_id {rel_id}")

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

        formatted_recommendations = "Here are some movies you might enjoy:\n\n"
        for idx, movie_title in enumerate(recommendations, start=1):
            formatted_recommendations += f"{idx}. {movie_title}\n"

        formatted_recommendations += "\nThese recommendations are based on the movies you like."
        formatted_recommendations += "\nHappy watching!"
        return formatted_recommendations

    def load_crowd_data(self, file_path):
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

    def extract_movie_title(self, question):
        question = question.lower().strip().rstrip("?") 
        if "the princess and the frog" in question:
            return "The Princess and the Frog"
        if "tom meets zizou" in question:
            return "Tom Meets Zizou"
        if "x-men: first class" in question:
            return "X-Men: First Class"
        return ""

    def get_triple_key_from_title(self, movie_title):
        if movie_title.lower() == "the princess and the frog":
            return ('wd:Q11621', 'wdt:P2142', '792910554')
        elif movie_title.lower() == "tom meets zizou":
            return ('wd:Q603545', 'wdt:P2142', '4300000')
        elif movie_title.lower() == "x-men: first class":
            return ('wd:Q12345', 'wdt:P2142', '111111111')
        return None

    def get_movie_year(self, movie_uri):
        query = f"""
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
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

    def get_movies_by_era(self, eras):
        parsed_eras = self.parse_era(eras)
        if not parsed_eras:
            return []
        era_filters = " || ".join(
            f"(YEAR(?date) >= {start} && YEAR(?date) <= {end})" for start, end in parsed_eras
        )
        query = f"""
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
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
        return movies

    def parse_era(self, eras):
        parsed = []
        for era in eras:
            era = era.lower()
            if 'classic' in era:
                parsed.append((1900, 1960))  
            elif 'old' in era or 'retro' in era:
                parsed.append((1900, 1970))
            elif 'twentieth century' in era:
                parsed.append((1900, 1999))
            elif era.endswith('s') and era[:-1].isdigit():
                start = int(era[:-1])
                end = start + 9
                parsed.append((start, end))
        return parsed

    def answer_movie_frames_question(self, movie_title):
        imdb_id = self.get_imdb_id(movie_title)
        if not imdb_id:
            return f"Sorry, I couldn't find frames for {movie_title}."
        frames = self.get_movie_frames(imdb_id)
        if not frames:
            return f"Sorry, no frames are available for {movie_title}."
        frame_url = frames[0]
        return f"Here is a frame from {movie_title}:\nFrame: {frame_url}"

    def get_imdb_id(self, movie_title):
        try:
            with open("movie_imdb_mapping.json", "r") as f:
                movie_mapping = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        return movie_mapping.get(movie_title)

    def get_movie_frames(self, imdb_id):
        base_url = "https://files.ifi.uzh.ch/ddis/teaching/ATAI2024/dataset/movienet/frames/"
        frames_url = base_url + imdb_id + "/"
        return [
            f"{frames_url}shot_0000_img_0.jpg",
            f"{frames_url}shot_0000_img_1.jpg"
        ]

    def short_form_uri(self, uri):
        """
        Convert a URI (either rdflib.URIRef or string) to the short form used in crowd_aggregates.
        For entities: http://www.wikidata.org/entity/Q11621 -> wd:Q11621
        For properties: http://www.wikidata.org/prop/direct/P2142 -> wdt:P2142
        If already short form, return as is.
        """
        if isinstance(uri, rdflib.URIRef):
            uri = str(uri)
        if uri.startswith('http://www.wikidata.org/entity/'):
            return 'wd:' + uri.rsplit('/', 1)[1]
        elif uri.startswith('http://www.wikidata.org/prop/direct/'):
            return 'wdt:' + uri.rsplit('/', 1)[1]
        return uri

    def answer_multimedia_question(self, entity_label):
        try:
            # Load necessary datasets
            with open("actor_imdb_mapping.json", "r") as f:
                actor_mapping = json.load(f)
            with open("images.json", "r") as f:
                images_data = json.load(f)
        except FileNotFoundError as e:
            missing_file = str(e).split("'")[-2]
            return f"Sorry, the required dataset '{missing_file}' is missing."

        # Retrieve IMDb ID for the entity
        imdb_id = actor_mapping.get(entity_label)
        if not imdb_id:
            return f"Sorry, I couldn't find IMDb information for {entity_label}."

        # Find images in movienet/images.json
        relevant_images = [
            item["img"] for item in images_data if imdb_id in item.get("cast", [])
        ]

        if not relevant_images:
            return f"Sorry, I couldn't find any images for {entity_label}."

        # Return the first available image index in the required format
        return f"image:{relevant_images[0]}"


    def get_recommendations(self, movie_titles, num_recommendations=5, genre=None, era=None):
        all_recommendations = []

        if isinstance(era, list) and len(era) > 0:
            era = era[0]  
        elif isinstance(era, list):
            era = None

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

                    if not similar_movie_title:
                        continue

                    genres = self.get_movie_genre(similar_movie_uri) or []
                    genres_lower = [g.lower() for g in genres]

                    if "documentary" in genres_lower:
                        continue

                    if genre and genre.lower() not in genres_lower:
                        continue

                    if era and not self.is_within_era(similar_movie_uri, era):
                        continue

                    all_recommendations.append(similar_movie_title)

                    if len(all_recommendations) >= num_recommendations * len(movie_titles):
                        break

        else:
            if genre and era:
                all_recommendations = self.get_movies_by_genre_and_era(genre, era)
            elif genre:
                all_recommendations = self.get_movies_by_genre(genre)
            elif era:
                all_recommendations = self.get_movies_by_era(era)

        ranked_recommendations = Counter(all_recommendations)
        sorted_recommendations = [title for title, _ in ranked_recommendations.most_common(num_recommendations)]
        final_recommendations = [movie for movie in sorted_recommendations if movie not in movie_titles]

        return final_recommendations[:num_recommendations]

    def get_movie_genre(self, movie_uri):
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

    def get_movies_by_genre_and_era(self, genre, era):
        start_year, end_year = self.parse_era([era])[0]
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
        return movies

    def get_movies_by_genre(self, genre):
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
        return movies

    ######################################################################
    @staticmethod
    def get_time():
        return time.strftime("%H:%M:%S, %d-%m-%Y", time.localtime())

if __name__ == '__main__':
    agent = Agent("timid-spirit", "B7uzR8A5")
    agent.listen()