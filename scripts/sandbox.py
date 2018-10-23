import multiprocessing
from graphbispectrum.graph import Graph

n = 7

def f(i):
    return Graph.calculate_invariant_edges("bispectrum", n=n, i=i, sparse=True)


if __name__ == '__main__':
    iso_classes = {}
    e = int(round(n * (n - 1.0) / 2.0))
    p = multiprocessing.Pool(multiprocessing.cpu_count())
    for iso_classes_i in p.map(f, range(e + 1)):
        iso_classes.update(iso_classes_i)
    print "%d Isomorphism classes for graphs with %d vertices" % (len(iso_classes), n)