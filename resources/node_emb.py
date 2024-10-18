import pandas as pd 
import numpy as np
from sklearn import model_selection, preprocessing
from tensorflow.keras import layers, Model, optimizers, losses
import networkx as nx
from scipy.io import mmread
from stellargraph import StellarGraph, mapper, layer

def get_adj_matrix(
    matrix_path
):
    adj_matrix = mmread(matrix_path).todense()
    adj_matrix[range(adj_matrix.shape[0]), range(adj_matrix.shape[1])] = 0
    
    return adj_matrix

def get_network(
    adj_matrix
):
    network = nx.from_numpy_array(adj_matrix)

    return network

def get_edges(
    network
):
    edges = pd.DataFrame(network.edges)
    edges.columns = ['source', 'target']

    return edges

def get_node_features(
    ed_emb_path
):
    node_features = pd.read_csv(ed_emb_path)

    return node_features

def get_node_subjects(
    input_path
):
    node_subjects = pd.read_csv(input_path)

    return node_subjects

def create_graph(
    nodes_features,
    edges
):
    graph = StellarGraph(nodes = nodes_features, edges = edges)
    print(graph.info())

    return graph