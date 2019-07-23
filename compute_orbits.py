from analysis.utils.graph_analysis import *
import pandas as pd

bs6 = pd.read_pickle('analysis/output/bispectrum_on_6_nodes')
orbit_assignment, graphs_by_orbit, bs_matches, bs_distance = find_bispectrum_matches(bs6)

OA = pd.DataFrame({'orbit_assignment': orbit_assignment})
GBO = pd.DataFrame({'graphs_by_orbit': graphs_by_orbit})
BSM = pd.DataFrame({'bs_matches': bs_matches})
BSD = pd.DataFrame({'bs_distance': bs_distance})

OA.to_pickle('analysis/output/6n_orbit_assignment')
GBO.to_pickle('analysis/output/6n_graphs_by_orbit')
BSM.to_pickle('analysis/output/6n_bs_matches')
BSD.to_pickle('analysis/output/6n_bs_distance')