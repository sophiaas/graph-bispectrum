import multiprocessing
import sys
from graphbispectrum.graph import Graph

try:
    n = int(sys.argv[-1])
except:
    n = 7

def f(i):
    return i, Graph.calculate_invariant_edges("bispectrum", n=n, i=i, sparse=True)


if __name__ == '__main__':
    iso_classes = {}
    e = int(round(n * (n - 1.0) / 2.0))
    p = multiprocessing.Pool(multiprocessing.cpu_count())
    for i, iso_classes_i in p.map(f, range(e + 1)):
        print "\t %d isomorphism classes for graphs with %d vertices and %d edges" % (len(iso_classes_i), n, i)
        iso_classes.update(iso_classes_i)
    print "---"
    print "%d isomorphism classes for graphs with %d vertices" % (len(iso_classes), n)