import scipy as sp
import numpy as np
import csv
import os
import sys
import re
from queue import Queue
from pathlib import Path
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import networkx as nx
import pickle

from typing import Union


def check_symmetry(a:np.array, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def load_source_dataset(corpus_path:Union[str,Path], keys_path:Union[str,Path]) -> dict:
    """
    Loads the English Bible dataset from the Kaggle dump and returns a dictionary with it and the keys mappings
    WARNING, this works for small datasets as it loads everything in memory
    
    Args:
        corpus_path (Path|str): Path to the entire Kaggle bible corpus
        keys_path (Path|str): Path to the key mappings of the bible corpus
    """
    key_verse_map = {}
    
    with open(corpus_path, newline='') as f:
        corpus = csv.reader(f)
        corpus_db = list(corpus)  # make a list out of the iterator, this is the entire DB
        verses = [r[-1] for r in corpus_db[1:]]  # omit the first line; It's the header
        
    with open(keys_path, newline='') as f:
        rows = csv.reader(f)
        kvs = list(rows)
        key_verse_map = { i[0] : i[1] for i in kvs[1:] } # omit the first line; It's the header
        
    return key_verse_map, corpus_db 


def embed_compare(txt:list(str), model, n_close:int=21, algorithm:str='inner') -> tuple(np.array):
    """
    Computes the sentence encoding of each of the strings in the input list
    Computes the similarity between them

    Args:
        txt (list(str)): list of strings, each one will be embedded into a single vector
        model (tf.model): tensorflow model Universal Sentence Encoder
        n_close (int): number of closest elements to extract from the similarity matrix
        algorithm (str): [inner|cosine] Algorithm to compute the similarity between vectors
    """
    bible_embeddings = model(verses)
    similarity_matrix_inner = np.inner(bible_embeddings, bible_embeddings)
    assert(check_symmetry(similarity_matrix_inner))
    
    # get the closest and farthest ~ N for each
    # https://www.kite.com/python/answers/how-to-find-the-n-maximum-indices-of-a-numpy-array-in-python
    # https://numpy.org/doc/stable/reference/generated/numpy.argpartition.html
    # https://numpy.org/doc/stable/reference/generated/numpy.partition.html
    n = n_close
    n = 21  # such as n>1 , when n==1 it shows only self-similarity
    partitions = np.argpartition(similarity_matrix_inner, -n, axis=0)
    n_close_ids = partitions[-n:]
    # farthest = partitions[:n]
    # now there are 2 arrays, each with verses number of elements, each array contains 
    n_close_ids = np.array(n_close_ids).transpose()
    # n_far = np.array(farthest).transpose()
    assert(n_close_ids.shape == (bible_embeddings.shape[0],n))
    # somehow this is not sorted as it should
    n_close_distances = np.array([similarity_matrix_inner[i][n_close_ids[i]] for i in range(similarity_matrix_inner.shape[0])])
    assert(n_close_distances.shape == (bible_embeddings.shape[0],n))
    
    return bible_embeddings, n_close_ids, n_close_distances


def create_db_dict(bible_embeddings:np.array, corpus_db: list, key_verse_map:dict, 
                   n_close:np.array, n_close_distances:np.array) -> Tuple(dict, nx.Graph):
    """
    Creates a bible database in dictionary format for easy indexing

    Args:
        bible_embeddings (np.array): [description]
        corpus_db (list): [description]
        key_verse_map (dict): [description]

    Returns:
        dict: a dict database for quick indexing
        nx.Graph: a nx.Graph containing the entire input domain
    """
        # db contains all the information AND the embeddings, this also contains the graph information
    # db contains all the information AND the embeddings, this also contains the graph information
    bible_db = {}

    graph_dict = {}

    for i in range(1, len(corpus_db)-1):
        verse = corpus_db[i]
    #     k_id = int(verse[0])
        v_idx = int(i-1)  # force int because pyvis complains about this
        val = {
            'index':v_idx,
            'id': int(verse[0]),
            'name': f"{key_verse_map[verse[1]]} {verse[2]}:{verse[3]}",
            'book_id': int(verse[1]),
            'chapter_id': int(verse[2]),
            'verse_id': int(verse[3]),
            'text': verse[4],
            'embedding': bible_embeddings[i],
            'close_to': n_close[i],  # ids
            'close_to_distance': n_close_distances[i], 
        }
        
        bible_db[v_idx] = val
        # now compute the graph for networkx  # force int because pyvis complains about this
        graph_dict[v_idx] = {int(k):1/v  for k,v in zip(n_close[i], close_matrix[i])}
        
    g = nx.Graph(graph_dict)
    return bible_db, g


def main():
    """ 
    Open sources and define DBs
    Save DB pickle dumps
    """
    #TODO get conf, env or input variables for the moment these are hardcoded
    
    # loading data
    BASE_PATH_DB = Path("/home/leo/projects/AI/Datasets/text/religion/bible/kaggle-bible-corpus")
    
    keys_path = BASE_PATH_DB / "key_english.csv"
    corpus_path = BASE_PATH_DB / "t_asv.csv"
    
    # TF Universal Sentence Encoder models
    #@title Load the Universal Sentence Encoder's TF Hub module
    MODELS_BASE_PATH = "/home/leo/projects/AI/Datasets/Tensorflow/tf-hub/"
    # module_url = os.path.join(BASE_PATH, "universal-sentence-encoder-lite_2")
    module_url = os.path.join(MODELS_BASE_PATH, "universal-sentence-encoder-multilingual_3")
    # module_url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
    # module_url = "https://tfhub.dev/google/universal-sentence-encoder-lite/2"
    # module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    # module_url = 'https://tfhub.dev/google/universal-sentence-encoder-large/5'
    
    BIBLE_DB_PATH = "../db/bible-db.pkl"
    BIBLE_EMBEDDINGS_PATH = "../db/bible-embeddings.pkl"

    similarity_algorithm = 'inner'

    # load bible dataset
    key_verse_map, corpus_db = load_source_dataset(corpus_path, keys_path)
    # load tensorflow Universal Embedding 
    model = hub.load(module_url)
    # print ("module %s loaded" % module_url)

    # compute embedding vectors and get similarities
    bible_embeddings, n_close_ids, n_close_distances = embed_compare(txt, model, algorithm='inner')

    # create database in a dictionary for fast indexing
    bible_db, g_nx = create_db_dict(bible_embeddings, corpus_db, key_verse_map)
    
    # Save databases
    # save the DB ## kind of big, 86 mb
    with open(BIBLE_DB_PATH, 'wb') as f:  
        pickle.dump(bible_db, f, pickle.HIGHEST_PROTOCOL)

    # the embeddings only
    with open(BIBLE_EMBEDDINGS_PATH, 'wb') as f:
        pickle.dump(bible_embeddings, f, pickle.HIGHEST_PROTOCOL)

    # the NetworkX graph
    with open(networkx_graph_db_path, 'wb') as f:
        pickle.dump(g_nx, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
    