import itertools
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
# from funcy import join
import networkx as nx
# from node2vec import Node2Vec
# from GraphWave.graphwave import graphwave
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import os
from pylab import rcParams
import math
from analysis.utils.graphwave.graphwave import graphwave

"""
RUN IN PYTHON 2.7

Example usage:
>>> map_idx = 3
>>> G, node_list = gen_state_space(map_idx)
>>> vecs, pca_coords, node_labels, cluster_dict, cluster_centers = learn_graphwave(G, k=5)
# >>> print_clusters(G, cluster_dict)
>>> plot_graph_clusters(G, cluster_dict, pca_coords, pos_type="cluster", jitter=True, node_labels=True, arrows=True)

Note that the color gradient that maps nodes roughly corresponds to the distance of the clusters in k-space

To plot the subgraph centered around a particular node, call plot_graph_clusters() with isolate_node=node_number.
"""
def gen_plots(map_idx, k=5, n_components=5, motifs=True, motif_degree=None, node_labels=True, jitter=True, pos_type="cluster", cmap="rainbow", arrows=True, save_name="graph"):
    G = gen_state_space(map_idx)
    vecs, pca_coords, cluster_centers, cluster_dict = learn_graphwave(G, k=k)
    save_dir = "plots/"+save_name+"/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plot_graph_clusters(G, cluster_dict, pca_coords, pos_type=pos_type, jitter=jitter, node_labels=node_labels, arrows=arrows, cmap=cmap, save_fig=True, save_name=save_dir+"full.png")
    plt.close()
    if motifs:
        for c in cluster_dict.keys():
            for n in cluster_dict[c]:
                plot_graph_clusters(G, cluster_dict, pca_coords, isolate_node=n, motif_degree=motif_degree, pos_type=pos_type, jitter=jitter, node_labels=node_labels, cmap=cmap, arrows=arrows, save_fig=True, save_name=save_dir+"cluster"+str(c)+"_node"+str(n)+".png", node_size=200, jitter_degree=10)
                plt.close()
#

def gen_hierarchy(state_embeddings, G, ll_transition_matrix, lights, hierarchy_depth, action_penalty=-.01):
    cluster_dicts = {}
    cluster_centers = {}
    cluster_lights = {}
    n_nodes = len(state_embeddings)
    hierarchy = {}
    for d in range(hierarchy_depth):
        k = 3 ** (d+1)
        km = KMeans(n_clusters=k)
        km.fit(state_embeddings)
        xy_pca = PCA(n_components=2)
        pca_coords = xy_pca.fit_transform(StandardScaler().fit_transform(state_embeddings))
        node_labels = km.labels_
        hierarchy[d] = node_labels
        cluster_dict = {}
        cluster_center = {}
        cluster_light = {}
        for c in np.unique(node_labels):
            nodes = np.where(node_labels == c)[0]
            cluster_dict[c] = nodes
            l = []
            for n in nodes:
                if G.nodes(data=True)[n]['coords'] in lights and G.nodes(data=True)[n]['light_on'] == 1:
                    l.append(G.nodes(data=True)[n]['coords'])
            cluster_light[c] = len(set(l))
            coords = [pca_coords[a] for a in nodes]
            cluster_center[c] = tuple(np.mean(coords, axis=0))
        cluster_dicts[d] = cluster_dict
        cluster_centers[d] = cluster_center
        cluster_lights[d] = cluster_light
    hierarchy[d+1] = np.array(range(len(state_embeddings)))
    cluster_dicts[d+1] = {x: np.array([x]) for x in range(n_nodes)}
    cluster_centers[d+1] = {i: tuple(x) for i, x in enumerate(pca_coords)}
    bin_hierarchy = {}
    for d in hierarchy.keys():
        bin_hierarchy[d] = ["{0:010b}".format(x) for x in hierarchy[d]]
    stacked = np.vstack(hierarchy.values()).T
    vecs = []
    for row in stacked:
        vecs.append(list(row))
    binary_stacked = np.vstack(bin_hierarchy.values()).T
    binary_vecs = []
    for row in binary_stacked:
        binary_vecs.append("".join(list(row)))
    transition_matrices = {}
    reward_matrices = {}
    graph_abstractions = {}
    for d in range(hierarchy_depth):
        A, transition_matrix, reward_matrix = gen_graph_abstraction(G, hierarchy[d], action_penalty)
        transition_matrices[d] = transition_matrix
        reward_matrices[d] = reward_matrix
        graph_abstractions[d] = A
    transition_matrices[d+1] = ll_transition_matrix
    return hierarchy, vecs, binary_vecs, graph_abstractions, transition_matrices, reward_matrices, cluster_lights, cluster_dicts

# for idx, a in enumerate(graph_abstractions.values()):
#     plot_graph_abstraction(a, cluster_centers[idx], "abstraction"+'_'+str(idx), -2, 2, -2, 2)
#     plt.close()


def learn_graphwave(G, k=5, n_components=5):
    """
    Learns node embeddings via GraphWave and clusters the embeddings into k clusters using KMeans.

    Input
    -----
    G: networkx graph
        any graph (multiedge, directed OK)
    k: int
        number of clusters
    n_components: int
        dimensionality of reduced embedding to cluster

    Returns
    -------
    vecs: np.array
        learned vector embeddings
    pca_coords: np.array
        nodes projected onto first 2 principle components, for plotting in x,y space
    cluster_centers: dict
        cluster numbers are keys, values are x,y coords of the mean of x,y coords of nodes in each cluster
    cluster_dict: dict
        cluster numbers are keys, values are lists of nodes in each cluster
    """
    vecs, heat_print, taus = graphwave(G, "automatic")
    pca = PCA(n_components=n_components)
    trans_data = pca.fit_transform(StandardScaler().fit_transform(vecs))
    xy_pca = PCA(n_components=2)
    pca_coords = xy_pca.fit_transform(StandardScaler().fit_transform(vecs))
    km = KMeans(n_clusters=k)
    km.fit(trans_data)
    node_labels = km.labels_
    centroids = km.cluster_centers_
    # centroids = []
    # for c in np.unique(node_labels):
    # 	nodes = np.where(node_labels == c)[0]
    # 	coords = [pca_coords[a] for a in nodes]
    # 	centroids.append(np.mean(coords, axis=0))
    min_centroid = np.argmin(np.mean(centroids, axis=0))
    centroid_dists = {cosine(centroids[min_centroid], a): idx for idx, a in enumerate(centroids)}
    centroids_sorted = np.sort(centroid_dists.keys())
    relabel = {}
    for idx, val in enumerate(centroids_sorted):
        relabel[centroid_dists[val]] = idx
    node_labels = [relabel[x] for x in list(node_labels)]
    cluster_dict = {}
    cluster_centers = {}
    for c in np.unique(node_labels):
        nodes = np.where(node_labels == c)[0]
        cluster_dict[c] = nodes
        coords = [pca_coords[a] for a in nodes]
        cluster_centers[c] = tuple(np.mean(coords, axis=0))
    return vecs, pca_coords, node_labels, cluster_dict, cluster_centers

def learn_node2vec(G, k=5, p=.1, q=2, dimensions=64, walk_length=30, num_walks=200, workers=4, window=8, min_count=1, batch_words=4):
    """
    Learns node embeddings via node2vec and clusters the embeddings into k clusters using KMeans.
    """
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers, p=p, q=q)
    model = node2vec.fit(window=window, min_count=min_count, batch_words=batch_words)
    vecs = np.zeros((len(G.nodes()), dimensions))
    for idx, node in enumerate(G.nodes()):
        vecs[idx] = model.wv.get_vector(str(node))
        codebook, distortion = scipy.cluster.vq.kmeans(vecs, k)
        code, dist = scipy.cluster.vq.vq(vecs, codebook)
        cluster_dict = {}
    for c in np.unique(code):
        nodes = np.where(code == c)[0]
        cluster_dict[c] = nodes
    return cluster_dict

def print_clusters(G, cluster_dict):
    """
    For printing node data for items in learned clusters
    """
    for c in cluster_dict.keys():
        print(str(c)+":")
        for a in cluster_dict[c]:
            print(str(a)+" - "+str(G.nodes[a]))


def gen_graph_abstraction(G, hierarchy, action_penalty=-.01):
    actions = [1, 2, 3, 4, 5]
    attribute_matrix = nx.attr_matrix(G, edge_attr="action")[0]
    edge_sums = {}
    action_prob = {}
    n_clust = len(range(0,np.max(hierarchy)+1))
    transition_matrix = [np.zeros((n_clust, n_clust)) for a in actions]
    reward_matrix = [np.zeros((n_clust, n_clust)) + action_penalty for a in actions]
    # for a in actions:
        # trans_prob[(a)] = np.zeros((len(cluster_dict.keys()), len(cluster_dict.keys())))
    for x in range(0,np.max(hierarchy)+1):
        if x in hierarchy:
            x_idx = np.array([i for i,j in enumerate(hierarchy) if j==x])
            xes = attribute_matrix[x_idx]
            print(xes)
            for a in actions:
                action_prob[(x, a)] = np.sum(xes==a) / float(len(xes))
                for y in range(0,np.max(hierarchy)+1):
                    if y in hierarchy:
                        y_idx = np.array([i for i,j in enumerate(hierarchy) if j==y])
                        if len(y_idx) > 0:
                            xy = attribute_matrix[x_idx][:, y_idx]
                            transition_matrix[a-1][x, y] = np.sum(xy==a)
                            if a == 1:
                                summed_reward = np.sum(xy==a)
                                if summed_reward > 0:
                                    reward_matrix[a-1][x, y] = summed_reward
                # norms0 = np.sum(transition_matrix[a-1], axis=0)
                # norms1 = np.sum(transition_matrix[a-1], axis=1)
                # # nonzero0 = norms0 > 0
                # nonzero1 = norms1 > 0
                # # transition_matrix[a-1][:, nonzero0] /= norms0[nonzero0][None, :]
                # transition_matrix[a-1][nonzero1] /= norms1[nonzero1][:, None]

    transition_matrix = {a: transition_matrix[a-1] for a in actions}
    reward_matrix = {a: reward_matrix[a-1] for a in actions}
#     for k in transition_matrix.keys():
#         for x, xval in enumerate(transition_matrix[k]):
#             if (x, k) in action_prob:
#                 transition_matrix[k][x] *= action_prob[(x, k)]
#             else:
#                 action_prob[(x, k)] = 0
    edge_list = []
    for action_idx, action_matrix in enumerate(transition_matrix.values()):
        trans = action_matrix.nonzero()
        edge_list.extend([(trans[0][i], trans[1][i], {"action": actions[action_idx], "weight": transition_matrix[action_idx+1][trans[0][i], trans[1][i]], "reward": reward_matrix[action_idx+1][trans[0][i], trans[1][i]]}) for i, item in enumerate(trans[0])])
    A = nx.MultiDiGraph()
    A.add_edges_from(edge_list)
    return A, transition_matrix, reward_matrix

# def gen_graph_abstraction(G, cluster_dict):
#     actions = [1, 2, 3, 4, 5]
#     attribute_matrix = nx.attr_matrix(G, edge_attr="action")[0]
#     edge_sums = {}
#     action_prob = {}
#     transition_matrix = [np.zeros((len(cluster_dict.keys()), len(cluster_dict.keys()))) for a in actions]
#     # for a in actions:
#         # trans_prob[(a)] = np.zeros((len(cluster_dict.keys()), len(cluster_dict.keys())))
#     for x in cluster_dict.keys():
#         x_idx = np.array([i for i,j in enumerate(list(G.nodes())) if j==x])
#         xes = attribute_matrix[x_idx]
#         for a in actions:
#             action_prob[(x, a)] = np.sum(xes==a) / float(len(xes))
#             for y in cluster_dict.keys():
#                 y_idx = np.array([i for i,j in enumerate(list(G.nodes())) if j==y])
#                 xy = attribute_matrix[x_idx][:, y_idx]
#                 transition_matrix[a-1][x, y] = np.sum(xy==a)
#             norms = np.sum(transition_matrix[a-1], axis=1)
#             nonzero = norms > 0
#             transition_matrix[a-1][nonzero] /= norms[nonzero][:, None]
#     transition_matrix = {a: transition_matrix[a-1] for a in actions}
#     for       k in transition_matrix.keys():
#         for x, xval in enumerate(transition_matrix[k]):
#             transition_matrix[k][x] *= action_prob[(x, k)]
#     edge_list = []
#     for action_idx, action_matrix in enumerate(transition_matrix.values()):
#         trans = action_matrix.nonzero()
#         edge_list.extend([(trans[0][i], trans[1][i], {"action": actions[action_idx], "weight": transition_matrix[action_idx+1][trans[0][i], trans[1][i]]}) for i, item in enumerate(trans[0])])
#     A = nx.MultiDiGraph()
#     A.add_edges_from(edge_list)
#     return A, transition_matrix

def plot_graph_abstraction(A, cluster_centers, save_name="abstraction", x_min=None, x_max=None, y_min=None, y_max=None):
    save_dir = "plots/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cmapx = plt.get_cmap("rainbow")
    x = np.linspace(0, 1, np.max(A.nodes()) + 1)
    colors = [cmapx(xx) for xx in x]
    if x_min is None:
        x_min = np.min([x[0] for x in cluster_centers.values()])
        x_max = np.max([x[0] for x in cluster_centers.values()])
        y_min = np.min([x[1] for x in cluster_centers.values()])
        y_max = np.max([x[1] for x in cluster_centers.values()])
    x_range = abs(x_max - x_min)
    y_range = abs(y_max - y_min)
    x_padding = x_range / 5.0
    y_padding = y_range / 5.0
    x_min -= x_padding
    x_max += x_padding
    y_min -= y_padding
    y_max += y_padding
    ax = plt.axis([x_min, x_max, y_min, y_max])
    axis_ratio = x_range / y_range
    plt.rcParams["figure.figsize"] = (np.max([15, 15*axis_ratio]), np.min([15, 15*axis_ratio]))
    nx.draw_networkx(A, arrows=True, node_color=colors, with_labels=True, pos=cluster_centers, node_size=200, font_size=6)
    plt.savefig(save_dir+save_name+".png")

def plot_graph_clusters(G, cluster_dict, pca_coords=None, node_labels=True, jitter=True, isolate_node=None, motif_degree=None, highlight_node=None, pos_type="grid", cmap="rainbow", arrows=True, save_fig=False, save_name=None, jitter_degree=30, node_size=150):
    """
    Input:
        G: networkx graph
        cluster_dict: dict
            output of learn_graphwave() or learn_node2vec()
        pca_coords: dict
            output of learn_graphwave(). Projections of node embeddings onto first 2 principle components. Pass with pos_type="cluster" to plot in PCA space.
        node_labels: bool
            if True, each node is labeled with its number
        jitter: bool
            if True, nodes are jittered from true coords to help view multiple nodes that lie on the same coordinate
        isolate_node: int
            may send an integer corresponding to a node in the graph to plot only that node and its neighbors
        highlight_node: int
            may send an integer corresponding to a node in the graph to change the color of that node to black. Useful for finding particular nodes of interest. default=None
        pos_type: str
            may be "grid" or "cluster". If "grid", plots nodes on an x,y grid. If cluster, superimposes nodes of the same cluster to the same point.
        cmap: str
            choose cmap
        arrows: bool
            turn on or off arrows on directed edges
        save_fig, save_name: bool, str
            for saving plot
    """
    colors = [0]*len(G.nodes())
    positions = [0]*len(G.nodes())
    cmapx = plt.get_cmap(cmap)
    x = np.linspace(0, 1, np.max(cluster_dict.keys()) + 1)
    col = [cmapx(xx) for xx in x]
    for c in cluster_dict.keys():
        for a in cluster_dict[c]:
            colors[a] = col[c]
    if highlight_node is not None:
        colors[highlight_node] = "black"
    node_pos = {}
    if pos_type == "grid":
        for i, n in enumerate(G.nodes(data=True)):
            node_pos[i] = [n[1]["y"], n[1]["x"]]
    elif pos_type == "cluster":
        if pca_coords is not None:
            node_pos = {a: pca_coords for a in G.nodes()}
            for i, a in enumerate(G.nodes()):
                node_pos[a] = pca_coords[i]
        else:
            grid_size = int(math.ceil(np.sqrt(len(cluster_dict.keys()))))
            coords = list(itertools.product(range(grid_size), repeat=2))
            for i, c in enumerate(cluster_dict.keys()):
                for a in cluster_dict[c]:
                    node_pos[a] = (coords[i][0], coords[i][1])
    else:
        raise ValueError("Invalid pos_type.")
    x_min = np.min([x[0] for x in node_pos.values()])
    x_max = np.max([x[0] for x in node_pos.values()])
    y_min = np.min([x[1] for x in node_pos.values()])
    y_max = np.max([x[1] for x in node_pos.values()])
    x_range = abs(x_max - x_min)
    y_range = abs(y_max - y_min)
    x_padding = x_range / 5.0
    y_padding = y_range / 5.0
    x_min -= x_padding
    x_max += x_padding
    y_min -= y_padding
    y_max += y_padding
    ax = plt.axis([x_min, x_max, y_min, y_max])
    axis_ratio = x_range / y_range
    plt.rcParams["figure.figsize"] = (np.max([15, 15*axis_ratio]), np.min([15, 15*axis_ratio]))
    if jitter:
        x_jitter = x_range / float(jitter_degree)
        y_jitter = y_range / float(jitter_degree)
        for a in G.nodes():
            node_pos[a] = (node_pos[a][0]+np.random.uniform(-x_jitter, x_jitter), node_pos[a][1]+np.random.uniform(-y_jitter, y_jitter))
    if isolate_node is not None:
        neighbors = list(G.neighbors(isolate_node)) + [isolate_node]
        if motif_degree is not None:
            next_neighbors = []
            for i in range(motif_degree):
                for n in neighbors:
                    next_neighbors.extend(list(G.neighbors(n)))
            neighbors = list(np.unique(next_neighbors))
        neighbors.sort()
        H = G.subgraph(neighbors)
        colors = [colors[a] for a in neighbors]
        nx.draw_networkx(H, arrows=arrows, node_color=colors, with_labels=node_labels, pos=node_pos, node_size=node_size, font_size=6)
    else:
        nx.draw_networkx(G, arrows=arrows, node_color=colors, with_labels=node_labels, pos=node_pos, node_size=node_size, font_size=6)
    if save_fig:
        if save_name is None:
            plt.savefig("clusters.png")
        else:
            plt.savefig(save_name)
    return col

def extract_map_features(game_map):
    """
    Extracts relevant features from game_map.

    Parameters
    ----------
    game_map: lst
        List of dictionaries specifying height and light information for each
        tile on the game board.

    Returns
    -------
    height, light, light_idx, num_lights, x, y: lsts
        Lists of dictionaries specifying the relevant features.
    x_dim, y_dim: ints
        Specify x and y dimensions of the game board.
    """
    shape = np.shape(game_map)
    height = np.zeros(shape, dtype=int)
    light = np.zeros(shape, dtype=int)
    light_idx = list(np.zeros(shape[0]*shape[1], dtype=int))
    for i in range(shape[0]):
        for j in range(shape[1]):
            height[i, j] = game_map[i][j]["h"]
            if game_map[i][j]["t"] == "l":
                light[i, j] = 1
    height = make_dict_list([val for sublist in height for val in sublist], "height")

    num_lights = np.sum(light, dtype=int)
    light_location = [val for sublist in list(light) for val in sublist]
    L = [val for sublist in np.nonzero(light_location) for val in sublist]
    for i in range(num_lights):
        light_idx[L[i]] = i + 1
    light = make_dict_list([val for sublist in list(light) for val in sublist], "light")
    light_idx = make_dict_list(light_idx, "light_idx")

    x = [list(np.arange(shape[1])) for i in range(shape[0])]
    x = make_dict_list([val for sublist in x for val in sublist], "x")

    y = [list(np.repeat(i, shape[1])) for i in range(shape[0])][::-1]
    y = make_dict_list([val for sublist in y for val in sublist], "y")

    x_dim = shape[1]
    y_dim = shape[0]

    return height, light, light_idx, num_lights, x, y, x_dim, y_dim

def transitions(state_space, actions, x_dim, y_dim, num_lights):
    """
    Returns a transition matrix for the state space and set of actions.

    Parameters
    ----------
    state_space: lst
        List of dictionaries defining all possible states for a level.
    actions: lst
        Set of possible actions.
    x_dim, y_dim: ints
        Dimensions of the game board.
    num_lights: int
        Number of light tiles on the game board.

    Returns
    -------
    transition_matrix: csr_matrix
        Sparse transition matrix of shape num_actions x num_states x num_states
    """
    transition_matrix = [lil_matrix((len(state_space), len(state_space))) for i in actions]

    for i, state_1 in enumerate(state_space):
        for j, state_2 in enumerate(state_space):
            for a, action in enumerate(actions):

                # light action
                if action == 1:
                    if (state_1["board_properties"] == state_2["board_properties"]) and (state_1["direction"] == state_2["direction"]):
                        if state_1["board_properties"]["light"] == 1 and state_2["board_properties"]["light"] == 1:
                            if abs(state_2["light_on"] - state_1["light_on"]) == 1:
                                transition_matrix[a][i , j] = 1

                # jump action
                if action == 2:
                    if state_1["light_on"] == state_2["light_on"]:
                        height_diff = state_2["board_properties"]["height"] - state_1["board_properties"]["height"]
                        if height_diff == 1 or height_diff < 0:
                            if state_1["direction"] == state_2["direction"]:
                                if state_1["direction"] == 0:
                                    if (state_1["board_properties"]["y"] > 0) and (state_2["board_properties"]["x"] == state_1["board_properties"]["x"]) and (state_2["board_properties"]["y"] == (state_1["board_properties"]["y"] - 1)):
                                        transition_matrix[a][i , j] = 1
                                if state_1["direction"] == 1:
                                    if ((state_1["board_properties"]["x"] + 1) <= x_dim) and (state_2["board_properties"]["y"] == state_1["board_properties"]["y"]) and (state_2["board_properties"]["x"] == (state_1["board_properties"]["x"] + 1)):
                                        transition_matrix[a][i, j] = 1
                                if state_1["direction"] == 2:
                                    if ((state_1["board_properties"]["y"] + 1) <= y_dim) and (state_2["board_properties"]["x"] == state_1["board_properties"]["x"]) and (state_2["board_properties"]["y"] == (state_1["board_properties"]["y"] + 1)):
                                        transition_matrix[a][i, j] = 1
                                if state_1["direction"] == 3:
                                    if (state_1["board_properties"]["x"] > 0) and (state_2["board_properties"]["y"] == state_1["board_properties"]["y"]) and (state_2["board_properties"]["x"] == (state_1["board_properties"]["x"] - 1)):
                                        transition_matrix[a][i, j] = 1

                # walk action
                if action == 3:
                    if state_1["light_on"] == state_2["light_on"]:
                        if state_1["board_properties"]["height"] == state_2["board_properties"]["height"]:
                            if state_1["direction"] == state_2["direction"]:
                                if state_1["direction"] == 0:
                                    if (state_1["board_properties"]["y"] > 0) and (state_2["board_properties"]["x"] == state_1["board_properties"]["x"]) and (state_2["board_properties"]["y"] == (state_1["board_properties"]["y"] - 1)):
                                        transition_matrix[a][i, j] = 1
                                if state_1["direction"] == 1:
                                    if ((state_1["board_properties"]["x"] + 1) <= x_dim) and (state_2["board_properties"]["y"] == state_1["board_properties"]["y"]) and (state_2["board_properties"]["x"] == (state_1["board_properties"]["x"] + 1)):
                                        transition_matrix[a][i, j] = 1
                                if state_1["direction"] == 2:
                                    if ((state_1["board_properties"]["y"] + 1) <= y_dim) and (state_2["board_properties"]["x"] == state_1["board_properties"]["x"]) and (state_2["board_properties"]["y"] == (state_1["board_properties"]["y"] + 1)):
                                        transition_matrix[a][i, j] = 1
                                if state_1["direction"] == 3:
                                    if (state_1["board_properties"]["x"] > 0) and (state_2["board_properties"]["y"] == state_1["board_properties"]["y"]) and (state_2["board_properties"]["x"] == (state_1["board_properties"]["x"] - 1)):
                                        transition_matrix[a][i, j] = 1

                # turn_r action
                if action == 4:
                    if state_1["light_on"] == state_2["light_on"]:
                        if state_1["board_properties"] == state_2["board_properties"]:
                            if state_2["direction"] == ((state_1["direction"] - 1) % 4):
                                transition_matrix[a][i, j] = 1

                # turn_l action
                if action == 5:
                    if state_1["light_on"] == state_2["light_on"]:
                        if state_1["board_properties"] == state_2["board_properties"]:
                            if state_2["direction"] == ((state_1["direction"] + 1) % 4):
                                transition_matrix[a][i, j] = 1

    return transition_matrix

def self_transitions(transition_matrix, state_space, actions):
    """
    Adds self-transitions to a transition_matrix for impossible actions.

    Parameters
    ----------
    transition_matrix: csr_matrix
        Sparse transition matrix
    state_space: lst
        List of dictionaries defining all possible states for a level.
    actions: lst
        Set of possible actions.

    Returns
    -------
    transition_matrix: csr_matrix
        Sparse transition matrix with self-transitions added.
    """
    for a, action in enumerate(actions):
        for i, state in enumerate(state_space):
            if np.sum(transition_matrix[a][i, :], axis=1) == 0:
                transition_matrix[a][i, i] = 1
        transition_matrix[a] = csr_matrix(transition_matrix[a])
    return transition_matrix

def make_dict_list(value, key):
    """
    Returns a list containing a dictionary with value assigned to key.
    """
    l = []
    for i in range(np.shape(value)[0]):
        d = {}
        d[key] = value[i]
        l.append(d)
    return l
