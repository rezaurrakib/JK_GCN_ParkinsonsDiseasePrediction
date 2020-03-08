import os
import numpy as np
import pandas as pd
import networkx as nx
import tensorflow as tf
import matplotlib.pyplot as plt

from ggplot import *
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from Jumping_Knowledge_Network.models.jk_att import *
from Jumping_Knowledge_Network.models.jk_lstm import *
from Jumping_Knowledge_Network.models.jk_concat import *
from Jumping_Knowledge_Network.models.jk_maxpool import *
from tensorflow.contrib.tensorboard.plugins import projector


def features_embedding_visualize(dense_features, all_labels):
    #vis_folder = "Performance/Result_WeightedCrossEntropyLoss/Learning_Rate_0.001/Hideen_Layers_Embedding/JK_LSTM/Perplex_50/"
    vis_folder = "Final_Layer_Embedding/GENDER/JK_LSTM/"
    if not os.path.exists(vis_folder):
        os.makedirs(vis_folder)

    transformed = TSNE(n_components=2, perplexity=50, n_iter=5000).fit_transform(dense_features)
    print("transformed.shape : ", transformed)

    colors = ['green', 'red', 'black']
    node_colors = []
    for i in range(dense_features.shape[0]):
        node_colors.append(colors[all_labels[i]])

    def label_printer(i):
        if i == 1:
            return "Class A"
        else:
            return "Class B"

    df = pd.DataFrame()
    df['tsne_x'] = transformed[:, 0]
    df['tsne_y'] = transformed[:, 1]
    df['label'] = all_labels
    df['label'] = df['label'].apply(label_printer)

    chart = ggplot(df, aes(x='tsne_x', y='tsne_y', color='label'))\
            + geom_point(size=70, alpha=0.95)\
            + ggtitle("Initial Feature Embedding for Network")

    chart.save(vis_folder + "Initial_feature_embedding.png")


def features_embedding_visualize_hidden_layer(dense_features, all_labels, fold, model_name):
    #vis_folder = "Performance/Result_WeightedCrossEntropyLoss/Learning_Rate_0.001/Hideen_Layers_Embedding/JK_LSTM/"
    vis_folder = "Final_Layer_Embedding/AVERAGE/JK_LSTM/"
    if not os.path.exists(vis_folder):
        os.makedirs(vis_folder)

    transformed = TSNE(n_components=2, perplexity=50, n_iter=5000).fit_transform(dense_features)
    print("transformed.shape : ", transformed)

    colors = ['green', 'red', 'black']
    node_colors = []
    for i in range(dense_features.shape[0]):
        node_colors.append(colors[all_labels[i]])

    def label_printer(i):
        if i == 1:
            return "Class A"
        else:
            return "Class B"

    df = pd.DataFrame()
    df['tsne_x'] = transformed[:, 0]
    df['tsne_y'] = transformed[:, 1]
    df['label'] = all_labels
    df['label'] = df['label'].apply(label_printer)

    chart = ggplot(df, aes(x='tsne_x', y='tsne_y', color='label'))\
            + geom_point(size=70, alpha=0.9)\
            + ggtitle("Hidden Layers Feature Embedding for " + model_name.upper() +  " Network")

    chart.save(vis_folder + model_name + "_hidden_feature_embedding_fold_" + str(fold+1) + ".png")


def features_embedding_visualize_hidden_layer_gender(dense_features, all_labels, gender_list, fold, model_name):
    # vis_folder = "Performance/Result_WeightedCrossEntropyLoss/Learning_Rate_0.001/Hideen_Layers_Embedding/JK_LSTM/"
    vis_folder = "Final_Layer_Embedding/GENDER/JK_LSTM/"
    if not os.path.exists(vis_folder):
        os.makedirs(vis_folder)

    transformed = TSNE(n_components=2, perplexity=50, n_iter=5000).fit_transform(dense_features)
    print("transformed.shape : ", transformed)

    colors = ['green', 'red', 'black']
    shapes = [21, 22, 23]
    node_colors = []
    node_shapes = []
    new_all_labes = []
    for i in range(324):
        if all_labels[i] == 1 and (gender_list[i] == 1 or gender_list[i] == 0):
            new_all_labes.append(0)
        if all_labels[i] == 1 and gender_list[i] == 2:
            new_all_labes.append(1)
        if all_labels[i] == 2 and (gender_list[i] == 1 or gender_list[i] == 0):
            new_all_labes.append(2)
        if all_labels[i] == 2 and gender_list[i] == 2:
            new_all_labes.append(3)

    for i in range(dense_features.shape[0]):
        node_colors.append(colors[all_labels[i]])
        node_shapes.append(shapes[gender_list[i]])

    def label_printer1(i):
        if i == 3:
            return "Class A"
        elif i == 2:
            return "Class A"
        elif i == 1:
            return "Class B"
        elif i == 0:
            return "Class B"

    def label_printer2(i):
        if i == 3:
            return "Class X"
        elif i == 2:
            return "Class Y"
        elif i == 1:
            return "Class X"
        elif i == 0:
            return "Class Y"

    df = pd.DataFrame()
    df['tsne_x'] = transformed[:, 0]
    df['tsne_y'] = transformed[:, 1]
    df['label1'] = new_all_labes
    df['label2'] = new_all_labes
    df['label1'] = df['label1'].apply(label_printer1)
    df['label2'] = df['label2'].apply(label_printer2)

    chart = ggplot(df, aes(x='tsne_x', y='tsne_y', color='label1', shape='label2')) \
            + geom_point(size=70, alpha=0.9, ) \
            + scale_color_manual(values=["#5CB0AA", "#EC7E7B"]) \
            + ggtitle("Hidden Layers Feature Embedding for " + model_name.upper() + " Network")

    chart.save(vis_folder + model_name + "_hidden_feature_embedding_fold_" + str(fold + 1) + ".png")


def euclidean_distance(f1, f2):
    diff = f1 - f2
    return np.sqrt(np.dot(diff, diff))


def affinity_visualize(adj, dense_features, all_labels, num_sample):
    graph = nx.Graph()
    num_nodes = dense_features.shape[0]
    print("num_nodes: ", num_nodes)
    c2 = [i for i in range(num_nodes) if all_labels[i] == 1]
    c3 = [i for i in range(num_nodes) if all_labels[i] == 2]
    c3 = c3[:75]

    print("c2: ", c2)
    print("c3: ", c3)
    idx = np.concatenate((c2, c3), axis=0)
    print("idx: ", idx)
    print("idx len : ", len(idx))
    dense_features = dense_features[idx, :]
    all_labels = [all_labels[item] for item in idx]
    print("all_labels: ", all_labels)
    #adj = adj[idx, :]
    #adj = adj[:, idx]
    num_nodes = len(idx)
    graph.add_nodes_from(np.arange(num_nodes))
    cnt = 0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj[i, j] != 0 and adj[j,i] != 0:
                cnt += 1
                graph.add_edge(i, j, weight=euclidean_distance(dense_features[i, :], dense_features[j, :]))

    print("No. of edges : ", cnt)
    node_colors = []
    colors = ['g', 'r', 'g']
    for i in range(num_nodes):
        node_colors.append(colors[all_labels[i]])
    nx.draw_networkx(graph, nx.spring_layout(graph, weight='weight', iterations=3, scale=1000), node_size=10, width=0.5,
                     node_color=node_colors, with_labels=False)
    #nx.draw_networkx(graph, nx.spectral_layout(graph), node_size=5, width=0.3, node_color=node_colors, with_labels=False)
    plt.show()
