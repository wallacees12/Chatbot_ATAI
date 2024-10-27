from speakeasypy import Speakeasy, Chatroom
from typing import List
import time
import rdflib
import os
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import spacy
import difflib
from spacy.pipeline import EntityRuler
import json

DEFAULT_HOST_URL = 'https://speakeasy.ifi.uzh.ch'
listen_freq = 2

class Agent:
    def __init__(self, username, password):
        self.username = username

        self.speakeasy = Speakeasy(host=DEFAULT_HOST_URL, username=username, password=password)
        self.speakeasy.login()
        self.graph = rdflib.Graph()
        self.graph.parse('14_graph.nt', format='turtle')

        self.label_to_entity = self.build_label_to_entity_dict()
        self.ent2lbl = {ent: str(lbl) for ent, lbl in self.graph.subject_objects(rdflib.RDFS.label)}
        self.relation_to_uri = self.build_relation_to_uri_dict()

        self.entity_emb, self.relation_emb, self.ent2id, self.id2ent, self.rel2id, self.id2rel = self.load_embeddings()
        self.nlp = spacy.load('en_core_web_md')
        self.nlp = self.add_movie_title_patterns(self.nlp)

    def add_movie_title_patterns(self, nlp):
        ruler = nlp.add_pipe('entity_ruler', before='ner')
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
                    room.post_messages(f'Hello! You can ask me factual questions about movies.')
                    room.initiated = True
                for message in room.get_messages(only_partner=True, only_new=True):
                    print(
                        f"\t- Chatroom {room.room_id} "
                        f"- new message #{message.ordinal}: '{message.message}' "
                        f"- {self.get_time()}")

                    question = message.message
                    response = self.process_question(question)
                    
                    room.post_messages(response)
                    room.mark_as_processed(message)
                    
            time.sleep(listen_freq)

    def process_question(self, question):
        doc = self.nlp(question)
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
                sparql_query = self.construct_sparql_query(entity_uri, relation_uri)
                factual_answers = self.execute_sparql_query(sparql_query)

                if not factual_answers:
                    embedding_answers = self.predict_with_embeddings(entity_uri, relation_uri)
                else:
                    embedding_answers = None

                response = self.get_response(entity_label, relation_label, factual_answers, embedding_answers)
            else:
                response = f"Sorry, I couldn't find information about '{entity_label}' or understand the relation '{relation_label}'."
        else:
            response = f"Sorry, I couldn't understand your question."

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

    def construct_sparql_query(self, entity_uri, relation_uri):
        query = f'''
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?answerLabel WHERE {{
            <{entity_uri}> <{relation_uri}> ?answer .
            ?answer rdfs:label ?answerLabel .
            FILTER (lang(?answerLabel) = "en")
        }}
        LIMIT 5
        '''
        return query

    def execute_sparql_query(self, query):
        results = self.graph.query(query)
        answers = [str(row['answerLabel']) for row in results]
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
        for idx in closest_ids[:5]:
            candidate_entity_uri = self.id2ent[idx]
            label = self.ent2lbl.get(candidate_entity_uri)
            if label:
                answers.append(label)
        return answers

    def get_response(self, entity_label, relation_label, factual_answers, embedding_answers):
        if factual_answers:
            answer_str = ', '.join(factual_answers)
            response = f"The {relation_label} of {entity_label} is {answer_str}.".encode('ascii', errors='replace').decode('ascii')
        elif embedding_answers:
            answer_str = ', '.join(embedding_answers)
            response = f"The answer suggested by embeddings: {answer_str}. (Embedding Answer)".encode('ascii', errors='replace').decode('ascii')
        else:
            response = f"Sorry, I couldn't find any information about the {relation_label} of {entity_label}."
        return response

    @staticmethod
    def get_time():
        return time.strftime("%H:%M:%S, %d-%m-%Y", time.localtime())

if __name__ == '__main__':
    agent = Agent("timid-spirit", "B7uzR8A5")
    agent.listen()
