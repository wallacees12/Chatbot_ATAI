# Movie Chatbot — ATAI

A conversational agent for movie question-answering built for the **Advanced Topics in AI (ATAI)** course at the University of Zurich. The bot connects to UZH's [Speakeasy](https://speakeasy.ifi.uzh.ch) chatroom platform and answers natural-language questions about films, actors, directors, and genres by querying a Wikidata-derived RDF knowledge graph.

<img src="chatbot_demo.gif" width="900"/>

## What it does

The bot routes incoming questions through four specialised pipelines:

| Pipeline       | Example                                                          | Backed by                                                  |
| -------------- | ---------------------------------------------------------------- | ---------------------------------------------------------- |
| **Factual**    | *Who directed Inception?*                                        | SPARQL over a 7M-triple RDF graph (`14_graph.nt`)          |
| **Embedding**  | *What's the genre of The Masked Gang: Cyprus?* (sparse coverage) | TransE entity + relation embeddings, cosine similarity     |
| **Crowdsource**| *Is it true that Spielberg directed Inception?*                  | Aggregated MTurk crowd labels with inter-rater agreement   |
| **Multimedia** | *Show me a picture of Pulp Fiction*                              | IMDb stills from the MovieNet dataset, ResNet-18 reranking |
| **Recommendation** | *Recommend movies like The Matrix*                           | Genre/era similarity over the KG                           |

## Architecture

```
user message
     │
     ▼
┌──────────────────┐
│ spaCy NER + rule │  ← entity extraction (movie titles, persons)
│  EntityRuler     │  ← top_10000_movies_by_votes.txt patterns
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ relation matcher │  ← verb_to_relation.json
│                  │  ← question_word_to_relation.json
└────────┬─────────┘
         │
   ┌─────┴──────┬─────────────┬──────────────┐
   ▼            ▼             ▼              ▼
SPARQL      Embeddings    Crowd data    Image pipeline
(rdflib)    (TransE)      (pandas agg)  (ResNet-18)
   │            │             │              │
   └────────────┴─────┬───────┴──────────────┘
                     ▼
              formatted reply
```

The full agent lives in [`TobyKh_Bot.py`](TobyKh_Bot.py). A minimal earlier version is in [`GoodBot.py`](GoodBot.py).

## Running the live bot

```bash
pip install -r requirements.txt   # speakeasypy, rdflib, spacy, torch, pandas, sklearn, colorama
python -m spacy download en_core_web_md
python TobyKh_Bot.py
```

> The bot expects `14_graph.nt` (the DDIS movie KG, ~7M triples) and the TransE embedding files (`entity_embeds.npy`, `relation_embeds.npy`) in the repo root. These are gitignored due to size — they ship with the ATAI course materials.

## CLI demo (no Speakeasy required)

`cli_demo.py` is a self-contained demo that mirrors the routing logic and visual style of the live bot, but reads from stdin and renders movie posters inline in the terminal via the [Kitty graphics protocol](https://sw.kovidgoyal.net/kitty/graphics-protocol/) (works in Ghostty, Kitty, WezTerm, iTerm2). It uses curated facts so it runs without the missing knowledge graph.

```bash
brew install chafa            # one-time, for inline image rendering
pip install Pillow            # one-time, optional
python3 cli_demo.py
```

Then ask things like:

```
who directed Inception?
recommend movies like The Matrix
show me a picture of Pulp Fiction
is it true that the director of Inception is Steven Spielberg?
```

Real movie posters are fetched once from Wikipedia and cached under `.poster_cache/`, so subsequent runs are fully offline.

## Files of note

| File | Purpose |
| ---- | ------- |
| `TobyKh_Bot.py`           | Full agent: NER, SPARQL, embeddings, crowd, multimedia |
| `GoodBot.py`              | Minimal Speakeasy + raw SPARQL agent |
| `SPARQL.py`               | Genre/era query helpers |
| `cli_demo.py`             | Self-contained terminal demo |
| `labels.json`             | Hardcoded entity & property → Wikidata ID overrides |
| `verb_to_relation.json`   | Verb lemma → KG relation mapping |
| `crowd_data.tsv`          | Raw MTurk crowd labels |
| `relation_embeds.npy`     | TransE relation embeddings (gitignored) |
| `actor_imdb_mapping.json` | Actor name → IMDb `nm` ID |
| `movie_imdb_mapping.json` | Movie title → IMDb `tt` ID |

## Acknowledgements

Built for the ATAI 2024 course at UZH. The knowledge graph, MovieNet image dataset, and crowd-source labels are course materials provided by the [DDIS group](https://www.ifi.uzh.ch/en/ddis.html).
