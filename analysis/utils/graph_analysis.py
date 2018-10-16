import numpy as np
import networkx as nx
import sklearn.cluster as cluster
from graphbispectrum import Function, Graph, Partition, SymmetricGroup
from analysis.utils.graph_generation import *
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def laplacians(graphs):
    """
    Generates normalized laplacian matrices for a list of graphs.
    """
    Ls = [nx.normalized_laplacian_matrix(G).todense() for G in graphs]
    return Ls

def make_sym_matrix(n, vals, dtype=int):
    """
    Generates a symmetric matrix from the values of the upper triangular of a matrix in vector form (vals).
    """
    m = np.zeros([n,n], dtype=dtype)
    xs,ys = np.triu_indices(n,k=1)
    m[xs,ys] = vals
    m[ys,xs] = vals
    return m

def get_upper_triangular(matrix, diagonal_offset=1):
    """
    Returns the upper triangular of a matrix.
    """
    xs, ys = np.triu_indices(matrix.shape[0], k=diagonal_offset)
    return matrix[xs, ys]

def compute_bispectrum(adj_mat, epsilon=1e-10):
    edges = get_upper_triangular(adj_mat)
    G = Graph.from_edges(edges)
    bispectrum = G.bispectrum()
    bispectrum_ = [np.array(bispectrum[i].todense()) for i in range(len(bispectrum))]
    bispectrum_ = [np.clip(x, a_min=epsilon, a_max=None) for x in bispectrum_]
    bispectrum_ = [x/np.sum(np.abs(x)) if np.sum(x) != 0 else x for x in bispectrum_]
    return bispectrum_

def bispectrum_on_n_nodes(n, save_checkpoints=True, save_dir='analysis/output/'):
    """
    Computes the bispectrum of all graphs on n nodes. 
    Returns results along with 2D PCA and TSNE in a pandas DataFrame.
    """
    print('generating all {} node graphs'.format(n))
    graphs = all_n_node_graphs(n)
    bs = []; gs = []
    if n < 5:
        print('generating orbits')
        orbit_num = []; perm = []
        orbits = find_isomorphisms(graphs)
        print('computing bispectrums for {} graphs'.format(len(graphs)))
        k = 0
        for i, o in enumerate(orbits):
            for j, g in enumerate(o):
                bispectrum = compute_bispectrum(g)
                bispectrum_ = np.hstack([x.ravel() for b in bispectrum for x in b])
                bs.append(bispectrum_)
                gs.append(g)
                perm.append(j)  
                orbit_num.append(i)
                if save_checkpoints:
                    df = pd.DataFrame({'graph': gs, 'orbit': orbit_num, 'bispectrum': bs, 'permutation': perm})
                    df.to_pickle(save_dir+'bispectrum_on_'+str(n)+'_nodes')
                print(k)
                k+=1
        df = pd.DataFrame({'graph': gs, 'orbit': orbit_num, 'bispectrum': bs, 'permutation': perm})
        
    else:
        print('computing bispectrums for {} graphs'.format(len(graphs)))
        k = 0
        for j, g in enumerate(graphs):
            bispectrum = compute_bispectrum(g)
            bispectrum_ = np.hstack([x.ravel() for b in bispectrum for x in b])
            bs.append(bispectrum_)     
            gs.append(g)
            if save_checkpoints:
                df = pd.DataFrame({'graph': gs, 'bispectrum': bs})
                df.to_pickle(save_dir+'bispectrum_on_'+str(n)+'_nodes')
            print(k)
            k+=1
        df = pd.DataFrame({'graph': gs, 'bispectrum': bs})
    df.to_pickle(save_dir+'bispectrum_on_'+str(n)+'_nodes')
    print('performing dimensionality reduction')
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    pca = PCA(n_components=2)
    tsne_results= tsne.fit_transform(bs)
    pca_results = pca.fit_transform(bs)
    df = df.merge(pd.DataFrame({'tsne_1': tsne_results[:,0], 'tsne_2': tsne_results[:,1], 
                           'pca_1': pca_results[:,0], 'pca_2': pca_results[:,1]}), 
                          left_index=True, right_index=True)
    df.to_pickle(save_dir+'bispectrum_on_'+str(n)+'_nodes')
    return df

def find_bispectrum_matches(df):
    bs_matches = np.zeros((len(df.bispectrum), len(df.bispectrum)))
    bs_distance = np.zeros((len(df.bispectrum), len(df.bispectrum)))
    for i, bs1 in enumerate(df.bispectrum):
        for j, bs2 in enumerate(df.bispectrum):
            diff = abs(bs1 - bs2)
            bs_distance[i, j] = np.sum(diff)
            if (diff < 1e-10).all():
                bs_matches[i, j] = 1
    m = [list(np.nonzero(x)[0]) for x in bs_matches]
    m.sort()
    m = list(orbits for orbits,_ in itertools.groupby(m))
    orbit_assignment = np.zeros(len(df.bispectrum), dtype=int)
    for idx, orbit in enumerate(m):
        for graph in orbit:
            orbit_assignment[graph] = idx
    graphs_by_orbit = [[df.graph[i] for i in n] for n in m]    
    return orbit_assignment, graphs_by_orbit, bs_matches, bs_distance
