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


def stylize_and_plot(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['bottom'].set_color(ppcolor2)
    ax.yaxis.grid(True, which='major', color=ppcolor2, ls='-', lw=1.5)
    pp.tick_params(axis='y', top='off', length=8, color=ppcolor1, direction='out')
    pp.tick_params(axis='x', top='off', length=8, color=ppcolor1, direction='out')
    for tick_label in ax.yaxis.get_ticklabels():
        tick_label.set_fontsize(12)
        tick_label.set_fontstyle('italic')
        tick_label.set_color(ppcolor1)
    for tick_label in ax.xaxis.get_ticklabels():
        tick_label.set_fontsize(12)
        tick_label.set_fontstyle('italic')
        tick_label.set_color(ppcolor1)
    pp.show()
    pp.clf()


#
# global configs
#

run_level = 31
ppcolor1 = '#000088'
ppcolor2 = '#ccffcc'


#
# read giant components
#

if run_level & 1 > 0:
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

if run_level & 2 > 0:
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

    ax = pp.subplot(111)
    ax.set_xlabel('Degree', color=ppcolor1, alpha=0.8)
    ax.set_ylabel('Frequency', color=ppcolor1, alpha=0.8)
    ax.loglog(all_dds['hamster'], ls='', marker='s', label='hamster')
    ax.loglog(all_dds['euroroad'], ls='', marker='o', label='euroroad')
    ax.loglog(all_dds['us-airports'], ls='', marker='d', label='us-airports')
    ax.loglog(all_dds['us-powergrid'], ls='', marker='p', label='us-powergrid')
    ax.legend(loc='upper right')
    stylize_and_plot(ax)
    print('done')


#
# assorted measurements
#

if run_level & 4 > 0:
    print('=> assorted measurements... ', end='', flush=True)

    table_data = np.zeros(7, dtype='a30, f4, f4, f4, f4')
    col_labels = ('Measurement',
                  'Net: Hamster',
                  'Net: Euroroad',
                  'Net: US Airports',
                  'Net: US Powergrid')

    table_data[0] = ('NumberOfNodes',
                     len(all_giants['hamster'].vs),
                     len(all_giants['euroroad'].vs),
                     len(all_giants['us-airports'].vs),
                     len(all_giants['us-powergrid'].vs))
    table_data[1] = ('MeanDegree',
                     '{:.4f}'.format(stat_moment(all_giants['hamster'], 1)),
                     '{:.4f}'.format(stat_moment(all_giants['euroroad'], 1)),
                     '{:.4f}'.format(stat_moment(all_giants['us-airports'], 1)),
                     '{:.4f}'.format(stat_moment(all_giants['us-powergrid'], 1)))
    table_data[2] = ('2ndStatMoment',
                     '{:.4f}'.format(stat_moment(all_giants['hamster'], 2)),
                     '{:.4f}'.format(stat_moment(all_giants['euroroad'], 2)),
                     '{:.4f}'.format(stat_moment(all_giants['us-airports'], 2)),
                     '{:.4f}'.format(stat_moment(all_giants['us-powergrid'], 2)))
    table_data[3] = ('TransitivityLocalAvg',
                     '{:.4f}'.format(all_giants['hamster'].transitivity_avglocal_undirected()),
                     '{:.4f}'.format(all_giants['euroroad'].transitivity_avglocal_undirected()),
                     '{:.4f}'.format(all_giants['us-airports'].transitivity_avglocal_undirected()),
                     '{:.4f}'.format(all_giants['us-powergrid'].transitivity_avglocal_undirected()))
    table_data[4] = ('TransitivityGlobal',
                     '{:.4f}'.format(all_giants['hamster'].transitivity_undirected()),
                     '{:.4f}'.format(all_giants['euroroad'].transitivity_undirected()),
                     '{:.4f}'.format(all_giants['us-airports'].transitivity_undirected()),
                     '{:.4f}'.format(all_giants['us-powergrid'].transitivity_undirected()))
    table_data[5] = ('ShortestPathAvg',
                     '{:.4f}'.format(shortest_path_avg(all_giants['hamster'])),
                     '{:.4f}'.format(shortest_path_avg(all_giants['euroroad'])),
                     '{:.4f}'.format(shortest_path_avg(all_giants['us-airports'])),
                     '{:.4f}'.format(shortest_path_avg(all_giants['us-powergrid'])))
    table_data[6] = ('Diameter',
                     all_giants['hamster'].diameter(directed=False),
                     all_giants['euroroad'].diameter(directed=False),
                     all_giants['us-airports'].diameter(directed=False),
                     all_giants['us-powergrid'].diameter(directed=False))

    pp.table(cellText=table_data, colLabels=col_labels, cellLoc='center', loc='center')
    pp.axis('tight')
    pp.axis('off')
    pp.show()
    pp.clf()
    print('done')


#
# local clustering distributions
#

if run_level & 8 > 0:
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

if run_level & 16 > 0:
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
