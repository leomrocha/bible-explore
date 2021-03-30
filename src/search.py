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


    
####
# DB graph transversal from the DB -> improve or pass the logic to NetworkX

# function to get the subgraph 

def _get_edges(n_index, node, close_points):
    """
    n_index: node index in the encoding matrix
    node: the node in the db
    close_points: the number of close points to get
    """
    nd_weight = zip(node['close_to'], node['close_to_distance'])
    nd_edges = []
    for n,w in nd_weight:
        if n != n_index:
            # distance is farthest the smaller it is, so changing it to make the reverse relation
            nd_edges.append((n_index, n, 1/w))
        if len(nd_edges) >= close_points:
            break
    return nd_edges


def get_subgraph(database, node_id:int, n_closest:int=5, n_depth:int=2):
    """
    Get the subgraph from a node id
    database: the entire bible database in a dict
    node_id: the node id
    close_points: the number of close points from each, => each point will have at most close_points outgoing edges
    levels: number of levels to go in depth for connections
    returns a networkx graph of the subgraph from the complete db centering the subgraph in the given node_id
    """
    # print(f"Getting sub graph node_id= {node_id} with {close_points} up to depth {levels}")
    g = nx.Graph()
    if node_id not in database:
        print(f"Node {node_id} not Found in the DB!")
        # there is no graph to build
        return g
        # TODO maybe return/raise an error/exception!?
    # Recursive is SO intuitive, but will explode the stack and memory for big graphs
    nid = node_id
    nodes_to_add = []  # (node_id, group, size, label, title, txt)
    edges_to_add = []  # (node_id, node_id, weight)
    discovered = set()
    q = Queue()
    q.put((node_id, 0))  # keep (node, depth from center in levels)
    cnt = 0
    # this tree/ graph transversal is not WOW HOW EFFICIENT, but it works well enough
    while not q.empty() and cnt < n_depth+1:
        node_id, lvl = q.get()
        discovered.add(node_id)
        node = database[node_id]
        nodes_to_add.append((node, lvl))
        # only add the edges if the level is not the max
        if lvl < n_depth:
            nd_edges = _get_edges(node_id, node, n_closest)
            for edg in nd_edges:
                edges_to_add.append((edg, lvl))
                sn, en, w = edg
                if en not in discovered:
                    q.put((en, lvl+1))
        cnt = lvl

    for node, lvl in nodes_to_add:
        g.add_node(node['index'], size=60/(lvl+1), group=node['book_id'], title=node['name'], data=node['text'])
    
    for edg, lvl in edges_to_add:
        (sn, en, w) = edg
        # print(edg)
        # pyvis complains that this are not int fields!! (but they are...)
        g.add_edge(int(sn), int(en), weight=w)
    # print("returning graph to caller", len(nodes_to_add), len(edges_to_add))
    return nodes_to_add, edges_to_add, g

# Search function
def get_closest_points(txt, model, embeddings, n=21, algorithm='inner'):
    """
    txt: the text to look for similarities
    n: the number of closest matches that will be searched
    algorithm: inner|cosine  # the algorithm to determine how the proximity is computed
    returns the closest n points to the input text based on the proximity algorithm
    """
    # compute input embedding 
    embd = model([txt])
    # compute proximity with all the existing points
    similarity = np.inner(embeddings, embd)
#     print(similarity.shape)
    # get the closest n points ids
    # such as n>1 , when n==1 it shows only self-similarity
    partitions = np.argpartition(similarity, -n, axis=0)
#     print(partitions.shape)
    n_close = partitions[-n:]
    # n_far = partitions[:n]
    # needs a complete matrix
    # return n_close, n_far
    return n_close

def depth_search(txt:str, model, embeddings, database:dict, n_closest=5, n_depth=3, algo='inner'):
    search_results = get_closest_points(txt, model, embeddings, n=n_closest, algorithm=algo).flatten()
    closest = search_results[0]
    nodes, edges, result_graph = get_subgraph(database, closest, n_closest=5, n_depth=3)
    return search_results, nodes, edges, result_graph
