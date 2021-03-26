from fastapi import FastAPI
# OS and file operations
import csv
import os
import sys
import re
from pathlib import Path
import pickle
# 
from functools import lru_cache

# models and NNs
# import scipy as sp
# import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import networkx as nx

# engine files
from . import config
from .search import depth_search

####
# Helpers and cache
@lru_cache()
def get_settings():
    cfg = config.cfg_factory()
    return cfg

#  Globals, these are good to get this way such as to use them as cached 
# settings = get_settings()
# # tensorflow model
# model = hub.load(settings.USE_MODULE_URL)
# # embeddings (numpy)
# with open(settings.BIBLE_EMBEDDINGS_PATH, 'rb') as f:
#     embeddings = pickle.load(f)
# # database (dict with all the details)
# with open(settings.BIBLE_DB_PATH, 'rb') as f:
#     database = pickle.load(f)


@lru_cache()
def _get_engines():
    settings = get_settings()
    # print("getting the settings from a .env file")
    # print(settings.schema(by_alias=True))
    # print(settings)
    # tensorflow model
    model = hub.load(settings.USE_MODULE_URL)
    # embeddings (numpy)
    with open(settings.BIBLE_EMBEDDINGS_PATH, 'rb') as f:
        embeddings = pickle.load(f)
    # database (dict with all the details)
    with open(settings.BIBLE_DB_PATH, 'rb') as f:
        database = pickle.load(f)
    return model, embeddings, database
    # return None, None, None


@lru_cache()
def _search(txt:str):
    # model, embeddings, database = _get_engines()
    settings = get_settings()
    model, embeddings, database = engines = _get_engines()
    results = depth_search(txt, model, embeddings, database, 
                         n_closest=settings.N_CLOSEST,
                         n_depth=settings.N_DEPTH,
                         algo=settings.ALGO)
    # print("Returning results ")
    # print(nodes, edges, result_graph)
    # return search_results, nodes, edges, result_graph
    return results

### 

app = FastAPI()

@app.get('/search')
async def search(query:str=""):
    settings = get_settings()
    # TODO if no query given, just select a random query from a basic list and return that
    search_results, nodes, edges, result_graph = _search(query)
    # print(f'results: {results}')
    # Can't return the result like that, something extra must be done to be able to encode in JSON!
    # maybe pass that to base64? or another encoding?
    return {"message": f'Hola Loca!! pediste {query}', 
            f"closest {settings.N_CLOSEST}": list(search_results.tolist()),
            # "nodes": nodes,  # TODO serialize
            # "edges": edges,  # TODO serialize
            "result_graph": None,  # NetworkX graph
            }