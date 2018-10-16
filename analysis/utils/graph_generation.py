import numpy as np
import itertools
from scipy.special import comb
from scipy.spatial.distance import squareform
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib
import math
import sklearn.cluster as cluster
from sklearn.decomposition import NMF

"""
Code for defining the orbits of small undirected subgraphs and finding the elements of those orbits within a larger undirected graph.

    >>> G = nx.fast_gnp_random_graph(12, .5)
    >>> relations, orbits = gen_relational_tensor(G, graph_size=4)
    >>> identify_isomorphic_subgraphs(G, relations, orbits)
    
"""

def all_n_node_graphs(graph_size):
    """
    Returns all symmetric n by n adjacency matrices.

    Parameters
    ----------
    graph_size: int
        size of the adjacency matrices

    Returns
    -------
    adj_mats: lst
        List of all n by n symmetric adjacency matrices.
    """
    nodes = np.arange(graph_size)
    configs = [tup for tup in itertools.product([0, 1], repeat=int(comb(len(nodes), 2)))]
    configs = sorted(configs, key=lambda x: sum(x))
    adj_mats = []
    for config in configs:
        t = squareform(config)
        if np.count_nonzero(np.sum(t, axis=0)) >= graph_size and np.count_nonzero(np.sum(t, axis=1)) >= graph_size:
            if graph_size == 2 or not np.array_equal(np.sum(t, axis=0), np.ones(graph_size)):
                adj_mats.append(t)
    return adj_mats

def transform(adj_mats):
    """
    Returns a list containing all permutations of the adjacency
    matrices in adj_mats.

    Parameters
    ----------
    adj_mats: lst
        Output of all_n_node_graphs

    Returns
    -------
    transforms: lst
        List the same length as adj_mats, where transforms[i] is a
        list containing all permutations of adj_mats[i]
    """
    n = np.arange(len(adj_mats[0]))
    perm = list(itertools.permutations(n))
    perm_rules = [list(zip(n, i)) for i in perm]
    transforms = []
    for mat in adj_mats:
        mat_transforms = []
        for rule in perm_rules:
            transform = mat.copy()
            for tup in rule:
                transform[:, tup[0]] = mat[:, tup[1]]
            ref = transform.copy()
            for tup in rule:
                transform[tup[0], :] = ref[tup[1], :]
            mat_transforms.append(transform)
        transforms.append(mat_transforms)
    return transforms

def gen_perm_mats(edge_size):
    """
    Returns a list containing all permutation edge_size x edge_size permutation matrices.

    Parameters
    ----------
    edge_size: int
        Desired edge size for the permutation matrices

    Returns
    -------
    perm_mats: lst
        List of all edge_size x edge_size permutation matrices.
    """
    n = np.arange(edge_size)
    perm = list(itertools.permutations(n))
    perm_rules = [list(zip(n, i)) for i in perm]
    perm_mats = []
    mat = np.identity(edge_size)
    for rule in perm_rules:
        perm_mat = mat.copy()
        for tup in rule:
            perm_mat[:, tup[0]] = mat[:, tup[1]]
        perm_mats.append(perm_mat)
    return perm_mats

def find_isomorphisms(adj_mats):
    """
    Groups all isomorphic graphs in a list of adjacency matrices. 
    Returns a list containing the non-redundant orbits.
    Tractable only for graphs with fewer than 5 nodes.

    Parameters
    ----------
    adj_mats: int
        List of adjacency matrices on n nodes

    Returns
    -------
    orbits: lst
        List of orbits. orbits[i] indexes orbit i and contains every element (permutation) of orbit i.
    """
    transforms = transform(adj_mats)
    match = np.zeros((len(adj_mats), len(adj_mats)))
    for i, mat_1 in enumerate(adj_mats):
        for j, mat_2 in enumerate(transforms):
            n = len([x for x in mat_2 if (x == mat_1).all()])
            if n > 0:
                match[i, j] = 1
    m = [list(np.nonzero(x)[0]) for x in match]
    m.sort()
    m = list(orbits for orbits,_ in itertools.groupby(m))
    orbits = [[adj_mats[i] for i in n] for n in m]
    return orbits

def gen_relational_tensor(graph, graph_size):
    """
    Returns a graph_size-D tensor, where the length of each dimension is equal to the number of nodes in graph. relations[a, b, ... , n] indexes an ordered graph_size subgraph (a < b < ... < n). The value of relations[a, b, ... , n] is an int which indexes orbits if that orbit[i] holds for that subgraph, or 0 otherwise. Also returns orbits, the list of orbit groups.

    Parameters
    ----------
    graph: nx.Graph
    graph_size: Size of isomorphic graphs to search for.

    Returns
    -------
    relations: graphs_size-D arr
        Tensor encoding which orbits hold for which ordered subgraphs of graph.
    orbits: lst
        List of orbits. Output of find_isomorphisms.
    """
    orbits = find_isomorphisms(graph_size)
    tensor_shape = tuple(np.repeat(len(graph.nodes()), graph_size))
    relations = np.zeros(tensor_shape)
    it = np.nditer(relations, flags=['multi_index'])
    while not it.finished:
        if not len(set(it.multi_index)) < len(it.multi_index): #no self-relations
            subgraph = graph.subgraph(list(it.multi_index))
            adj_mat = nx.adjacency_matrix(subgraph).todense()
            for idx, orbit in enumerate(orbits):
                for transformation in orbit:
                    if (adj_mat == transformation).all():
                        relations[it.multi_index] = idx
        it.iternext()
    return relations, orbits

def identify_isomorphic_subgraphs(graph, relations, orbits):
    """
    Plots graph and saves a .png for every orbit, where instances of each orbit are colored within the larger graph. 
    Takes a graph and the output of gen_relational_tensor as input.
    """
    all_orbits = []
    for idx, orbit in enumerate(orbits):
        orbit_nodes = []
        it = np.nditer(relations, flags=['multi_index'])
        while not it.finished:
            if it[0] == idx and not len(set(it.multi_index)) < len(it.multi_index):
                orbit_nodes.append(it.multi_index)
            it.iternext()
        all_orbits.append(orbit_nodes)
    for i, orbit in enumerate(all_orbits):
        for j, nodes in enumerate(orbit):
            orbit[j] = tuple(sorted(nodes))
            orbit.sort()
            all_orbits[i] = list(k for k,_ in itertools.groupby(orbit))
    for i, orbit in enumerate(all_orbits):
        if i is not 0:
            for j, nodes in enumerate(orbit):
                subgraph_edges = graph.subgraph(list(nodes)).edges()
                edge_colors = []
                edge_widths = []
                for edge in graph.edges():
                    if edge in subgraph_edges:
                        edge_colors.append(40 + 5*i)
                        edge_widths.append(2)
                    else:
                        edge_colors.append(0)
                        edge_widths.append(1)
                node_colors = []
                node_size = []
                for node in graph.nodes():
                    if node in nodes:
                        node_colors.append(40 + 5*i)
                        node_size.append(30)
                    else:
                        node_colors.append(0)
                        node_size.append(10)
                pos = nx.drawing.layout.circular_layout(graph)
                edge_vmax=40 + 5 * len(all_orbits)
                cmap = matplotlib.cm.get_cmap('OrRd')
                norm = matplotlib.colors.Normalize(vmin=0, vmax=edge_vmax)
                fig = plt.figure()
                fig.suptitle('orbit '+str(i), fontsize=14, color=cmap(norm(40 + 5*i)))
                nx.draw(graph, edge_cmap = plt.cm.OrRd, edge_vmin=0, edge_vmax=edge_vmax, cmap = plt.cm.OrRd, vmin=0, vmax=edge_vmax, node_color=node_colors, edge_color=edge_colors, pos=pos, width=edge_widths, node_size=node_size)
                fig.patch.set_facecolor('#D1D1D1')
                fig.savefig('orbit_'+str(i)+'_'+str(j)+'.png', facecolor=fig.get_facecolor())
                plt.close()
                
def plot_orbits(orbits, savefig=False, scale=2, n_cols=4):
    """
    Plots the output of find_isomorphisms()
    Each permutation in an orbit is plotted on the same axis.
    """
    for orbit_n, idxs in enumerate(orbits):
        N = len(idxs)
        cols = n_cols
        rows = int(math.ceil(np.float(N) / cols))
        gs = gridspec.GridSpec(rows, cols);
        fig = plt.figure(figsize=(cols*scale, rows*scale));
        for n, idx in enumerate(idxs):
            ax = fig.add_subplot(gs[n]);
            do_plot(idx, ax)
        fig.suptitle('orbit {}'.format(orbit_n), fontsize=20)
        if savefig:
            plt.savefig('orbit_'+str(orbit_n)+'.png')
            plt.close

def do_plot(idx, ax):
    """
    Helper function for plot_orbits.
    """
    G = nx.from_numpy_matrix(idx)
    pos = nx.drawing.layout.circular_layout(G)
    nx.draw(G, pos, ax, node_size=30, with_labels=False);
#     nx.draw_networkx(G, pos=pos, ax=ax, node_size=30, with_labels=False);    