import os
import matplotlib.pyplot as plt
import numpy as np

from .graph import Graph
from .symmetric_group import SymmetricGroup, Partition


FIGURES = os.path.join(os.path.dirname(__file__), "figures")

ns = [4, 5, 6, 7, 8]
num_graphs = 1
inter_inv = 1
inter_sep = 1

# 1st col check invariance 2nd col check separability
scores = np.zeros((len(ns), 2))

# elements = [(0, 1)]
elements = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
for ni, n in enumerate(ns):
    for i in xrange(num_graphs):
        g = Graph.erdos_renyi(n, edges=n/2)
        for j in xrange(inter_inv):
            g_ = g.apply_permutation(g.sn.random())
            # scores[ni, 0] += int(np.all([
            #     np.allclose(e1, e2) for e1, e2 in zip(g.power_spectrum(), g_.power_spectrum())]))
            scores[ni, 0] += int(np.all([np.allclose(
                g.bispectrum_element(*e).todense(),
                g_.bispectrum_element(*e).todense()
            ) for e in elements]))

        for j in xrange(inter_sep):
            h = Graph.erdos_renyi(n, edges=n/2)
            scores[ni, 1] += int(False in [np.allclose(
                g.bispectrum_element(*e).todense(),
                h.bispectrum_element(*e).todense()
            ) for e in elements])


plt.close('all')
plt.figure(0)
plt.title("Invariance and Separability scores vs N")
plt.plot(ns, scores[:,0]/(1.*num_graphs*inter_inv), c="r")
plt.plot(ns, scores[:,1]/(1.*num_graphs*inter_sep), c="b")
plt.savefig(os.path.join(FIGURES,
    "bis_scores__ns%d_ng%d_inv%d_sep%d_%s" % (
        len(ns), num_graphs, inter_inv, inter_sep,
        "_".join(("%s_%s" % (e1, e2) for e1, e2 in elements))
    )))
plt.show()