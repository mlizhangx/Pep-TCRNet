import networkx as nx
import pandas as pd
import numpy as np
from scipy.io import mmread
import itertools
import os

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import stellargraph as sg
from stellargraph import StellarGraph
from stellargraph import globalvar
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE

from stellargraph.data import UnsupervisedSampler
from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, feature_extraction, model_selection
from stellargraph import datasets
# from IPython.display import display, HTML
import matplotlib.pyplot as plt

EMBEDDING_FILENAME = '/wynton/home/zhang/leah/pred/graph/gnn/graphsage_semi_emb.csv'
TRAINED_PARAMS_DATA_FILE_PATH = '/wynton/home/zhang/leah/pred/graph/gnn/graph_semi_trained_weights.h5'
ADJ_MAT_DATA_FILE_PATH = '/wynton/home/zhang/leah/pred/graph/gnn/TCR_Sequence_Similarity_AdjacencyMatrix.mtx'
TCR_ANTIGEN_DATA_FILE_PATH = '/wynton/home/zhang/leah/pred/graph/gnn/tcr-antigen-latent-space.csv' 
TARGET_DATA_FILE_PATH = '/wynton/home/zhang/leah/pred/graph/gnn/feature_engineering_dat.csv'

adj_mat = mmread(ADJ_MAT_DATA_FILE_PATH)
adj_mat = adj_mat.todense()
adj_mat[range(adj_mat.shape[0]), range(adj_mat.shape[1])] = 0
Graph = nx.from_numpy_array(adj_mat)
edges = pd.DataFrame(Graph.edges)
edges.columns = ['source', 'target']
features = pd.read_csv(TCR_ANTIGEN_DATA_FILE_PATH).iloc[:, 1:]
target = pd.read_csv(TARGET_DATA_FILE_PATH)
target.columns = ['CDR3', 'y']
labels_raw = target['y'].unique().tolist()
class2idx = {labels_raw[k]: k for k in range(len(labels_raw))}
idx2class = {v: k for k, v in class2idx.items()}
node_subjects = target.y
nodes_features = features
G = StellarGraph(nodes = nodes_features, edges = edges)
train_subjects, test_subjects = model_selection.train_test_split(
    node_subjects, train_size=0.3, test_size=None,  random_state=42
)
target_encoding = preprocessing.LabelBinarizer()

train_targets = target_encoding.fit_transform(train_subjects)
test_targets = target_encoding.transform(test_subjects)
batch_size = 50
num_samples = [50, 50]
generator = GraphSAGENodeGenerator(G, batch_size, num_samples)
train_gen = generator.flow(train_subjects.index, train_targets, shuffle=True)
graphsage_model = GraphSAGE(
    layer_sizes=[30, 30], generator=generator, bias=True, dropout=0.15,
)
x_inp, x_out = graphsage_model.in_out_tensors()
prediction = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)
model = Model(inputs=x_inp, outputs=prediction)
model.compile(
    optimizer=optimizers.legacy.Adam(learning_rate=0.0005),
    loss=losses.categorical_crossentropy,
    metrics=["acc"],
)
test_gen = generator.flow(test_subjects.index, test_targets)
history = model.fit(
    train_gen, epochs=50, validation_data=test_gen, verbose=2, shuffle=False
)
test_metrics = model.evaluate(test_gen)
print("\nTest Set Metrics:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))
all_nodes = node_subjects.index
all_mapper = generator.flow(all_nodes)
all_predictions = model.predict(all_mapper)
embedding_model = Model(inputs=x_inp, outputs=x_out)
emb = embedding_model.predict(all_mapper)
embedding_model.save_weights(TRAINED_PARAMS_DATA_FILE_PATH)
print('Model saved successfully!')
pd.DataFrame(emb).to_csv(EMBEDDING_FILENAME)