from speakeasypy import Speakeasy, Chatroom
from typing import List
import os
import time
import rdflib
import nltk
from transformers import BertForTokenClassification, AutoTokenizer, AutoModel, pipeline
import torch
import gc
import psutil
import torch.multiprocessing as mp


# Download necessary resources for NLTK
nltk.download('punkt')

DEFAULT_HOST_URL = 'https://speakeasy.ifi.uzh.ch'
listen_freq = 2
# Disable tokenizers' parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set the multiprocessing start method
mp.set_start_method("spawn", force=True)

# Limit PyTorch threads to reduce memory usage
torch.set_num_threads(1)


def print_memory_usage():
    memory_info = psutil.virtual_memory()
    print(f"Memory usage: {memory_info.percent}%")

class Agent:
    def __init__(self, username, password):
        self.username = username
        
        # Initialize the Speakeasy Python framework and login.
        self.speakeasy = Speakeasy(host=DEFAULT_HOST_URL, username=username, password=password)
        self.speakeasy.login()  # This framework will help you log out automatically when the program terminates.
        
        # Load the RDF knowledge graph
        self.graph = rdflib.Graph()
        self.graph.parse('14_graph.nt', format='turtle')
        
        # Set up Huggingface models for NER and embeddings
        self.ner = pipeline(
            "ner",
            model=BertForTokenClassification.from_pretrained("prajjwal1/bert-tiny"),
            tokenizer=AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        )

        self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        self.model = AutoModel.from_pretrained("prajjwal1/bert-tiny").to(torch.device("cpu"))

        try:
            print("Testing NER...")
            print_memory_usage()
            test_result = self.ner("Who is the director of Inception?")
            print(f"NER Test Result: {test_result}")
            print_memory_usage()
        except Exception as e:
            print(f"An error occurred during NER: {e}")

        try:
            print("Testing embedding...")
            embedding = self.get_embedding("This is a test sentence.")
            print(f"Embedding: {embedding}")
            print_memory_usage()
        except Exception as e:
            print(f"An error occurred during embedding: {e}")

        # Free up memory manually
        del test_result, embedding
        
        torch.cuda.empty_cache()
        gc.collect()
    
            
    # Function to classify the question type
    def classify_question(self, question: str) -> str:
        tokens = nltk.word_tokenize(question.lower())
        
        if "director" in tokens or "screenwriter" in tokens or "released" in tokens:
            return "factual"
        elif "embedding" in tokens or "suggest" in tokens or "similarity" in tokens:
            return "embedding"
        else:
            return "unknown"

    # Function to extract the movie title using Huggingface NER
    def extract_movie_title(self, question: str) -> str:
        ner_results = self.ner(question)
        for entity in ner_results:
            if entity['entity'] == 'B-MOVIE':  # Assuming NER detects movie titles
                return entity['word']
        return ""

    # Function to generate SPARQL queries for factual questions
    def generate_sparql_query(self, question: str) -> str:
        if "director" in question.lower():
            movie_title = self.extract_movie_title(question)
            return f"""
            PREFIX ddis: <http://ddis.ch/atai/>
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            
            SELECT ?director WHERE {{
                ?movie rdfs:label "{movie_title}" .
                ?movie wdt:P57 ?directorItem .
                ?directorItem rdfs:label ?director .
            }}
            LIMIT 1
            """
        elif "screenwriter" in question.lower():
            movie_title = self.extract_movie_title(question)
            return f"""
            PREFIX ddis: <http://ddis.ch/atai/>
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            
            SELECT ?screenwriter WHERE {{
                ?movie rdfs:label "{movie_title}" .
                ?movie wdt:P58 ?screenwriterItem .
                ?screenwriterItem rdfs:label ?screenwriter .
            }}
            LIMIT 1
            """
        return None

    # Function to get sentence embeddings using BERT
    def get_embedding(self, sentence: str):
        inputs = self.tokenizer(sentence, return_tensors="pt")
        with torch.no_grad():
            embeddings = self.model(**inputs)
        return embeddings.last_hidden_state.mean(dim=1)

    # Function to handle embedding-based questions
    def embedding_based_answer(self, question: str, suggested_answer: str) -> str:
        question_embedding = self.get_embedding(question)
        suggested_answer_embedding = self.get_embedding(suggested_answer)
        
        # Calculate cosine similarity
        cosine_similarity = torch.nn.functional.cosine_similarity(question_embedding, suggested_answer_embedding)
        
        if cosine_similarity > 0.8:
            return f"Suggested answer: {suggested_answer} (Embedding Answer)"
        else:
            return f"Suggested answer: {suggested_answer} (Embedding Answer, lower confidence)"

    # Function to handle chatrooms and messages
    def listen(self):
        while True:
            rooms: List[Chatroom] = self.speakeasy.get_rooms(active=True)
            for room in rooms:
                if not room.initiated:
                    room.post_messages(f'Hello! This is a welcome message from {room.my_alias}.')
                    room.initiated = True

                # Retrieve new messages
                for message in room.get_messages(only_partner=True, only_new=True):
                    print(f"New message: '{message.message}'")
                    
                    # Classify the question type
                    question_type = self.classify_question(message.message)
                    
                    if question_type == "factual":
                        # Handle factual questions with SPARQL
                        sparql_query = self.generate_sparql_query(message.message)
                        if sparql_query:
                            result = self.graph.query(sparql_query)
                            response_message = "Query results:\n"
                            for row in result:
                                for item in row:
                                    if isinstance(item, rdflib.term.Literal):
                                        response_message += f"{item.value}\n"
                            room.post_messages(response_message.strip())
                    elif question_type == "embedding":
                        # Handle embedding-based questions
                        embedding_response = self.embedding_based_answer(message.message, "Richard Marquand")
                        room.post_messages(embedding_response)

                    # Mark the message as processed
                    room.mark_as_processed(message)

            time.sleep(listen_freq)

    @staticmethod
    def get_time():
        return time.strftime("%H:%M:%S, %d-%m-%Y", time.localtime())


if __name__ == '__main__':
    demo_bot = Agent("timid-spirit", "B7uzR8A5")  # Use your own credentials here
    demo_bot.listen()
