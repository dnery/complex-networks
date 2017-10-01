# igraph submodules
from igraph import Graph
from igraph import statistics

from os import path
from matplotlib import pyplot as pp

# import and config plaw
import powerlaw as pl
import numpy as np
np.seterr(divide='ignore', invalid='ignore')


#
# function definitions
#

def stat_moment(graph, which):
    measurement = 0
    for node in graph.vs:
        measurement += graph.degree(node) ** which
    return measurement / len(graph.vs)


def shortest_path_avg(graph):
    paths = graph.shortest_paths()
    n_paths = sum([len(x) for x in paths])
    n_steps = sum([sum(path) for path in paths])
    return n_steps / n_paths


def shortest_path_dist(graph):
    dims = graph.diameter()+1
    bins = np.linspace(0, dims, dims+1)
    freqs = np.zeros(dims, dtype=int)
    paths = graph.shortest_paths()
    for na in graph.vs:
        for nb in graph.vs:
            if na.index != nb.index:
                length = paths[na.index][nb.index]
                freqs[length] += 1
    return bins, freqs


#
# what's gonna run?
#

magic_number = 9


#
# read giant components
#

if magic_number & 1 > 0:
    print('=> reading networks... ', end='', flush=True)

    all_giants = {}
    current_path = path.join('networks', 'hamster.txt_awkd')
    with open(current_path, 'rb') as graph_file:
        graph = Graph.Read_Edgelist(graph_file, directed=False)
        all_giants['hamster'] = graph.components().giant()

    current_path = path.join('networks', 'euroroad.txt_awkd')
    with open(current_path, 'rb') as graph_file:
        graph = Graph.Read_Edgelist(graph_file, directed=False)
        all_giants['euroroad'] = graph.components().giant()

    current_path = path.join('networks', 'us-airports.txt_awkd')
    with open(current_path, 'rb') as graph_file:
        graph = Graph.Read_Ncol(graph_file, names=False, weights=True, directed=False)
        all_giants['us-airports'] = graph.components().giant()

    current_path = path.join('networks', 'us-powergrid.txt_awkd')
    with open(current_path, 'rb') as graph_file:
        graph = Graph.Read_Edgelist(graph_file, directed=False)
        all_giants['us-powergrid'] = graph.components().giant()

    print('done')

#
# degree distributions
#

if magic_number & 2 > 0:
    print('=> degree distributions, p-law alphas... ', flush=True)

    all_dds = {}
    all_dds['hamster'] = list(map(lambda x: x[2], all_giants['hamster'].degree_distribution().bins()))
    print('  hamster:\n\tpowerlaw alpha: {:.4f}'.format(pl.Fit(all_dds['hamster']).alpha))
    print('\t  igraph alpha: {:.4f}'.format(statistics.power_law_fit(all_dds['hamster']).alpha))

    all_dds['euroroad'] = list(map(lambda x: x[2], all_giants['euroroad'].degree_distribution().bins()))
    print('  euroroad:\n\tpowerlaw alpha: {:.4f}'.format(pl.Fit(all_dds['euroroad']).alpha))
    print('\t  igraph alpha: {:.4f}'.format(statistics.power_law_fit(all_dds['euroroad']).alpha))

    all_dds['us-airports'] = list(map(lambda x: x[2], all_giants['us-airports'].degree_distribution().bins()))
    print('  us-airports:\n\tpowerlaw alpha: {:.4f}'.format(pl.Fit(all_dds['us-airports']).alpha))
    print('\t  igraph alpha: {:.4f}'.format(statistics.power_law_fit(all_dds['us-airports']).alpha))

    all_dds['us-powergrid'] = list(map(lambda x: x[2], all_giants['us-powergrid'].degree_distribution().bins()))
    print('  us-powergrid:\n\tpowerlaw alpha: {:.4f}'.format(pl.Fit(all_dds['us-powergrid']).alpha))
    print('\t  igraph alpha: {:.4f}'.format(statistics.power_law_fit(all_dds['us-powergrid']).alpha))

    pp.title('degree distributions')
    p1, = pp.loglog(all_dds['hamster'], 's', label='hamster')
    p2, = pp.loglog(all_dds['euroroad'], 'o', label='euroroad')
    p3, = pp.loglog(all_dds['us-airports'], 'd', label='us-airports')
    p4, = pp.loglog(all_dds['us-powergrid'], 'p', label='us-powergrid')
    # pp.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
    pp.legend(handles=[p1, p2, p3, p4], loc=1)
    pp.grid(True)
    pp.show()
    pp.clf()

    print('done')


#
# assorted measurements
#

if magic_number & 4 > 0:
    print('=> assorted measurements... ', end='', flush=True)

    table_data = np.zeros((7, 4))
    col_labels = ('Hamster', 'Euroroad', 'US-airports', 'US-powergrid')
    row_labels = ('NumberOfNodes', 'MeanDegree', '2ndStatMoment',
                  'TransitivityLocalAvg', 'TransitivityGlobal',
                  'ShortestPathAvg', 'Diameter')

    table_data[0,0] = len(all_giants['hamster'].vs)
    table_data[0,1] = len(all_giants['euroroad'].vs)
    table_data[0,2] = len(all_giants['us-airports'].vs)
    table_data[0,3] = len(all_giants['us-powergrid'].vs)

    table_data[1,0] = '{:.4f}'.format(stat_moment(all_giants['hamster'], 1))
    table_data[1,1] = '{:.4f}'.format(stat_moment(all_giants['euroroad'], 1))
    table_data[1,2] = '{:.4f}'.format(stat_moment(all_giants['us-airports'], 1))
    table_data[1,3] = '{:.4f}'.format(stat_moment(all_giants['us-powergrid'], 1))

    table_data[2,0] = '{:.4f}'.format(stat_moment(all_giants['hamster'], 2))
    table_data[2,1] = '{:.4f}'.format(stat_moment(all_giants['euroroad'], 2))
    table_data[2,2] = '{:.4f}'.format(stat_moment(all_giants['us-airports'], 2))
    table_data[2,3] = '{:.4f}'.format(stat_moment(all_giants['us-powergrid'], 2))

    table_data[3,0] = '{:.4f}'.format(all_giants['hamster'].transitivity_avglocal_undirected())
    table_data[3,1] = '{:.4f}'.format(all_giants['euroroad'].transitivity_avglocal_undirected())
    table_data[3,2] = '{:.4f}'.format(all_giants['us-airports'].transitivity_avglocal_undirected())
    table_data[3,3] = '{:.4f}'.format(all_giants['us-powergrid'].transitivity_avglocal_undirected())

    table_data[4,0] = '{:.4f}'.format(all_giants['hamster'].transitivity_undirected())
    table_data[4,1] = '{:.4f}'.format(all_giants['euroroad'].transitivity_undirected())
    table_data[4,2] = '{:.4f}'.format(all_giants['us-airports'].transitivity_undirected())
    table_data[4,3] = '{:.4f}'.format(all_giants['us-powergrid'].transitivity_undirected())

    table_data[5,0] = '{:.4f}'.format(shortest_path_avg(all_giants['hamster']))
    table_data[5,1] = '{:.4f}'.format(shortest_path_avg(all_giants['euroroad']))
    table_data[5,2] = '{:.4f}'.format(shortest_path_avg(all_giants['us-airports']))
    table_data[5,3] = '{:.4f}'.format(shortest_path_avg(all_giants['us-powergrid']))

    table_data[6,0] = all_giants['hamster'].diameter(directed=False)
    table_data[6,1] = all_giants['euroroad'].diameter(directed=False)
    table_data[6,2] = all_giants['us-airports'].diameter(directed=False)
    table_data[6,3] = all_giants['us-powergrid'].diameter(directed=False)

    pp.table(cellText=table_data, rowLabels=row_labels, colLabels=col_labels, loc='center')
    pp.axis('tight')
    pp.axis('off')
    pp.show()
    pp.clf()

    print('done')


#
# local clustering distributions
#

if magic_number & 8 > 0:
    print('=> local clustering distributions...', end='', flush=True)

    bins = np.linspace(0, 1, 20)
    pp.title('local clustering distributions')
    pp.hist(([x for x in all_giants['hamster'].transitivity_local_undirected(mode='zero') if x > 0],
             [x for x in all_giants['euroroad'].transitivity_local_undirected(mode='zero') if x > 0],
             [x for x in all_giants['us-airports'].transitivity_local_undirected(mode='zero') if x > 0],
             [x for x in all_giants['us-powergrid'].transitivity_local_undirected(mode='zero') if x > 0]),
            bins, alpha=0.7, label=('hamster', 'euroroad', 'us-airports', 'us-powergrid'))
    pp.legend(loc=1)
    pp.grid(True)
    pp.show()
    pp.clf()

    print('done')


#
# shortest path distributions
#

if magic_number & 16 > 0:
    print('=> shortest path distributions...', end='', flush=True)

    all_pds = {}
    all_pds['hamster'] = shortest_path_dist(all_giants['hamster'])
    all_pds['euroroad'] = shortest_path_dist(all_giants['euroroad'])
    all_pds['us-airports'] = shortest_path_dist(all_giants['us-airports'])
    all_pds['us-powergrid'] = shortest_path_dist(all_giants['us-powergrid'])

    pp.title('shortest path distributions')
    p1, = pp.plot(all_pds['hamster'][1], lw=2.5, label='hamster')
    p2, = pp.plot(all_pds['euroroad'][1], lw=2.5, label='euroroad')
    p3, = pp.plot(all_pds['us-airports'][1], lw=2.5, label='us-airports')
    p4, = pp.plot(all_pds['us-powergrid'][1], lw=2.5, label='us-powergrid')
    pp.legend(handles=[p1, p2, p3, p4], loc=1)
    pp.grid(True)
    pp.show()
    pp.clf()

    print('done')
