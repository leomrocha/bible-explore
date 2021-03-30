# OS and file operations
import csv
import os
import sys
import re
from pathlib import Path
import pickle
# 
from functools import lru_cache
# Pydantic
from typing import List, Dict, Tuple, Optional, Sequence, Union

#
from fastapi import FastAPI, Query, Form

# UI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
# Error handling
from fastapi.responses import RedirectResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

# models and NNs
# import scipy as sp
# import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
# graph transversal and visualization
import networkx as nx
from pyvis.network import Network
# engine files
from . import config
from .search import depth_search, get_subgraph

# Define template directory
templates = Jinja2Templates(directory="src/templates") 


####
# Helpers and cache
@lru_cache()
def get_settings():
    cfg = config.cfg_factory()
    return cfg

# TODO unit test this function
@lru_cache
def _sanitize(query:str) -> str:
    query = query.lower()
    # TODO do some other validation if needed
    # replace all characters not in [a-zA-Z0-9 ,.:;_\-]  
    # and then replace all several spaces for only one
    return query


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


def is_book(query:str, books_idx:Dict[str, int]) -> Optional[int] :
    """[summary]
    Args:
        query (str): the query string 
        books_idx (Dict): a mapping containing the 

    Returns:
        Optional[int]: the book index 
    """
    # TODO try to match all the possible books:
    # [12] book [ chapter [verse] ]
    # TODO (correct orthographic errors -> give the hints in the UI to make it easier here) 
    # TODO verify the following regex for the book extraction (and write a unittest for it)
    # it's something like this:
    # ([123]?[ _\-.;:-]+\s*)?([a-zA-Z])+((\s+\d+\s*)?([ _\-.;:-]+\s*\d+)?)?
    # TODO compare the book in the query to the existing ones,
    # if not exists return None
    # else look for the chapter, if chapter not exist set chapter as 1
    # look for verse number, if verse not exists set verse as 1
    # return books_idx[book][chapter][verse]['index']
    
    # return 0
    return None
    # return verse_id, (book, chapter, verse)


@lru_cache()
def _search(txt:str):
    # model, embeddings, database = _get_engines()
    settings = get_settings()

    # to give the good result and save computation & memory 
    model, embeddings, database = engines = _get_engines()
    # TODO check if it is a book or verse already given by name
    closest = is_book(txt, database['book2key'])
    bible_db = database['db']
    if closest is not None:
        ret = get_subgraph(bible_db, closest, 
                           n_closest=settings.N_CLOSEST,
                           n_depth=settings.N_DEPTH)
        # print ("Returned Values: ", ret)
        # print("let s see \n \n ")
        # print("\n\n")
        nodes, edges, result_graph  = ret
        # no need to compute the similarity matrix, we know it already from the graph DB
        search_results = bible_db[closest]['close_to'][:settings.N_CLOSEST]
        results = search_results, nodes, edges, result_graph
    else:    
        results = depth_search(txt, model, embeddings, bible_db, 
                         n_closest=settings.N_CLOSEST,
                         n_depth=settings.N_DEPTH,
                         algo=settings.ALGO)
    # print("Returning results ")
    # print(nodes, edges, result_graph)
    # return search_results, nodes, edges, result_graph
    return results

### 


def nx2vis_dict(graph: nx.Graph) -> Dict[str, object]:
    """[summary]

    Args:
        graph (nx.Graph): [description]

    Returns:
        Dict[str, object]: [description]
    """
    # nt = Network(height='400px', width='50%', bgcolor='#2222', font_color='white')
    nt = Network(height='400px', width='100%')
    # nt = Network(height='100%', width='100%')
    nt.from_nx(graph)
    vis_nodes, vis_edges, vis_heading, vis_height, vis_width, vis_options = nt.get_network_data()
    d = {
        "nodes": vis_nodes, 
        "edges": vis_edges,
        "heading": vis_heading,
        "height": vis_height,
        "width": vis_width,
        "options": vis_options,
    }
    return d

################################################################
# App starts here
################################################################
app = FastAPI()
app.mount("/static", StaticFiles(directory="src/static"), name="static")


@app.get('/')
async def home():
    return templates.TemplateResponse("index.html", {"request": {}})


@app.get('/search', response_class=HTMLResponse)
# async def search(q: Optional[str] = Query(None, max_length=240, regex="HERE THE REGEX")):
# async def search(q: Optional[str] = Query(None, max_length=240)):
async def search(q: Optional[str] = Query("Genesis 1:1", max_length=240)):
    query = _sanitize(q)
    search_results, nodes, edges, result_graph = results = _search(query)
    # print(results[:2])
    visdict = nx2vis_dict(result_graph)
    return templates.TemplateResponse("index.html", 
                                      context={"request": {"q": q}, 
                                               "response":{
                                                "vis": visdict,
                                                "nodes":nodes }
                                                })

# in case anybody wants to play with the API, they'll go back to root
@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request, exc):
    return RedirectResponse("/")