# utils
import math as ma
from os import path
from sys import argv
from matplotlib import pyplot as pp

# igraph submodules
from igraph import Graph
from igraph import statistics

# import and config powerlaw
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


def shannon_diversity(graph):
    measurement = 0
    n_nodes = len(graph.vs)
    d_bins = graph.degree_distribution().bins()
    for freq in d_bins:
        if freq[2] > 0:
            pk = freq[2] / n_nodes
            measurement -= pk * ma.log2(pk)
    return measurement


def pearson_r(a, b):
    na = len(a)
    nb = len(b)
    if na != nb:
        raise ValueError('Samples differ in length')
    # standard sum
    sum_a = sum([float(x) for x in a])
    sum_b = sum([float(x) for x in b])
    # product sum
    sum_ab = sum([x * y for x, y in zip(a, b)])
    # squared sum
    sum_aa = sum([x ** 2 for x in a])
    sum_bb = sum([x ** 2 for x in b])
    # fraction members
    frac_a = na * sum_ab - sum_a * sum_b
    frac_b1 = (na * sum_aa - sum_a ** 2) ** .5
    frac_b2 = (na * sum_bb - sum_b ** 2) ** .5
    # final score value
    r_ab = frac_a / (frac_b1 * frac_b2)
    return r_ab


def shortest_path_avg(graph):
    paths = graph.shortest_paths()
    n_paths = sum([len(x) for x in paths])
    n_steps = sum([sum(path) for path in paths])
    return n_steps / n_paths


def shortest_path_dist(graph):
    dims = graph.diameter()+1
    bins = np.linspace(0, dims, dims+1)
    probs = np.zeros(dims, dtype=int)
    paths = graph.shortest_paths()
    for na in graph.vs:
        for nb in graph.vs:
            if na.index != nb.index:
                length = paths[na.index][nb.index]
                probs[length] += 1
    n_paths = sum([len(x) for x in paths])
    return bins, list(map(lambda x: x/n_paths, probs))


def cumulative_cci_dist(graph, bins):
    cci_values = graph.transitivity_local_undirected(mode='zero')
    cci_dist = np.zeros(len(bins), dtype=float)
    n_nodes = len(graph.vs)
    for i in range(len(bins)):
        cci_dist[i] = sum([1 for x in cci_values if x <= bins[i]])/n_nodes
    return cci_dist


def make_distribution(data, bins):
    norm_data = list(map(lambda x: x/max(data), data))
    dist = np.zeros(len(bins), dtype=float)
    inds = np.digitize(norm_data, bins)
    n_vals = len(norm_data)
    for ind in inds:
        dist[ind-1] += 1
    return list(map(lambda x: x/n_vals, dist))


def stylize_plot(ax, grid=True):
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.8)
    ax.spines['bottom'].set_color(ppcolor2)
    if grid:
        ax.yaxis.grid(True, which='major', color=ppcolor2, ls='-', lw=1.5)
    pp.tick_params(axis='y', top='off', length=8, color=ppcolor1, direction='out')
    pp.tick_params(axis='x', top='off', length=8, color=ppcolor1, direction='out')
    for tick_label in ax.yaxis.get_ticklabels():
        tick_label.set_fontsize(20)
        tick_label.set_fontstyle('italic')
        tick_label.set_color(ppcolor1)
    for tick_label in ax.xaxis.get_ticklabels():
        tick_label.set_fontsize(20)
        tick_label.set_fontstyle('italic')
        tick_label.set_color(ppcolor1)


#
# global configs
#

ppcolor1 = '#000088'
ppcolor2 = '#ccffcc'


#
# read giant components
#

print('reading networks... ', end='', flush=True)

all_graphs = {}
all_giants = {}
current_path = path.join('networks', 'hamster.txt_awkd')
with open(current_path, 'rb') as graph_file:
    graph = Graph.Read_Edgelist(graph_file, directed=False)
    all_graphs['hamster'] = graph
    all_giants['hamster'] = graph.components().giant()

current_path = path.join('networks', 'euroroad.txt_awkd')
with open(current_path, 'rb') as graph_file:
    graph = Graph.Read_Edgelist(graph_file, directed=False)
    all_graphs['euroroad'] = graph
    all_giants['euroroad'] = graph.components().giant()

current_path = path.join('networks', 'us-airports.txt_awkd')
with open(current_path, 'rb') as graph_file:
    graph = Graph.Read_Edgelist(graph_file, directed=False)
    all_graphs['us-airports'] = graph
    all_giants['us-airports'] = graph.components().giant()

current_path = path.join('networks', 'us-powergrid.txt_awkd')
with open(current_path, 'rb') as graph_file:
    graph = Graph.Read_Edgelist(graph_file, directed=False)
    all_graphs['us-powergrid'] = graph
    all_giants['us-powergrid'] = graph.components().giant()

print('done')

#
# degree distributions
#

print('degree distributions, p-law alphas', flush=True)

all_dds = {}
n_nodes = len(all_giants['hamster'].vs)
all_dds['hamster'] = list(map(lambda x: x[2]/n_nodes, all_giants['hamster'].degree_distribution().bins()))
print('├ hamster:\n│\tpowerlaw alpha: {:.4f}'.format(pl.Fit(all_dds['hamster']).alpha))
print('│\t  igraph alpha: {:.4f}'.format(statistics.power_law_fit(all_dds['hamster']).alpha))

n_nodes = len(all_giants['euroroad'].vs)
all_dds['euroroad'] = list(map(lambda x: x[2]/n_nodes, all_giants['euroroad'].degree_distribution().bins()))
print('├ euroroad:\n│\tpowerlaw alpha: {:.4f}'.format(pl.Fit(all_dds['euroroad']).alpha))
print('│\t  igraph alpha: {:.4f}'.format(statistics.power_law_fit(all_dds['euroroad']).alpha))

n_nodes = len(all_giants['us-airports'].vs)
all_dds['us-airports'] = list(map(lambda x: x[2]/n_nodes, all_giants['us-airports'].degree_distribution().bins()))
print('├ us-airports:\n│\tpowerlaw alpha: {:.4f}'.format(pl.Fit(all_dds['us-airports']).alpha))
print('│\t  igraph alpha: {:.4f}'.format(statistics.power_law_fit(all_dds['us-airports']).alpha))

n_nodes = len(all_giants['us-powergrid'].vs)
all_dds['us-powergrid'] = list(map(lambda x: x[2]/n_nodes, all_giants['us-powergrid'].degree_distribution().bins()))
print('└ us-powergrid:\n\tpowerlaw alpha: {:.4f}'.format(pl.Fit(all_dds['us-powergrid']).alpha))
print('\t  igraph alpha: {:.4f}'.format(statistics.power_law_fit(all_dds['us-powergrid']).alpha))

ax = pp.subplot(111)
ax.loglog(all_dds['hamster'], ls='', marker='s', label='hamster')
ax.loglog(all_dds['euroroad'], ls='', marker='o', label='euroroad')
ax.loglog(all_dds['us-airports'], ls='', marker='d', label='us-airports')
ax.loglog(all_dds['us-powergrid'], ls='', marker='p', label='us-powergrid')
ax.set_ylabel('Probability', color=ppcolor1, alpha=0.8, fontsize=20)
ax.set_xlabel('Degree', color=ppcolor1, alpha=0.8, fontsize=20)
ax.legend(loc='upper right', prop={'size': 20})
stylize_plot(ax)
pp.show()
pp.clf()
print('done')


#
# assorted measurements
#

print('assorted measurements... ', end='', flush=True)

table_data = np.zeros(8, dtype='f4, f4, f4, f4')
col_labels = ('Net: Hamster',
              'Net: Euroroad',
              'Net: US Airports',
              'Net: US Powergrid')
row_labels = ('Number of nodes',
              'Mean degree or <k>',
              '2nd stat moment or <k²>',
              'Avg local transitivity',
              'Global transitivity',
              'Avg shortest path length',
              'Diameter',
              'Entropy')

table_data[0] = (len(all_giants['hamster'].vs),
                 len(all_giants['euroroad'].vs),
                 len(all_giants['us-airports'].vs),
                 len(all_giants['us-powergrid'].vs))
table_data[1] = ('{:.4f}'.format(stat_moment(all_giants['hamster'], 1)),
                 '{:.4f}'.format(stat_moment(all_giants['euroroad'], 1)),
                 '{:.4f}'.format(stat_moment(all_giants['us-airports'], 1)),
                 '{:.4f}'.format(stat_moment(all_giants['us-powergrid'], 1)))
table_data[2] = ('{:.4f}'.format(stat_moment(all_giants['hamster'], 2)),
                 '{:.4f}'.format(stat_moment(all_giants['euroroad'], 2)),
                 '{:.4f}'.format(stat_moment(all_giants['us-airports'], 2)),
                 '{:.4f}'.format(stat_moment(all_giants['us-powergrid'], 2)))
table_data[3] = ('{:.4f}'.format(all_giants['hamster'].transitivity_avglocal_undirected()),
                 '{:.4f}'.format(all_giants['euroroad'].transitivity_avglocal_undirected()),
                 '{:.4f}'.format(all_giants['us-airports'].transitivity_avglocal_undirected()),
                 '{:.4f}'.format(all_giants['us-powergrid'].transitivity_avglocal_undirected()))
table_data[4] = ('{:.4f}'.format(all_giants['hamster'].transitivity_undirected()),
                 '{:.4f}'.format(all_giants['euroroad'].transitivity_undirected()),
                 '{:.4f}'.format(all_giants['us-airports'].transitivity_undirected()),
                 '{:.4f}'.format(all_giants['us-powergrid'].transitivity_undirected()))
table_data[5] = ('{:.4f}'.format(shortest_path_avg(all_giants['hamster'])),
                 '{:.4f}'.format(shortest_path_avg(all_giants['euroroad'])),
                 '{:.4f}'.format(shortest_path_avg(all_giants['us-airports'])),
                 '{:.4f}'.format(shortest_path_avg(all_giants['us-powergrid'])))
table_data[6] = (all_giants['hamster'].diameter(directed=False),
                 all_giants['euroroad'].diameter(directed=False),
                 all_giants['us-airports'].diameter(directed=False),
                 all_giants['us-powergrid'].diameter(directed=False))
table_data[7] = ('{:.4f}'.format(shannon_diversity(all_giants['hamster'])),
                 '{:.4f}'.format(shannon_diversity(all_giants['euroroad'])),
                 '{:.4f}'.format(shannon_diversity(all_giants['us-airports'])),
                 '{:.4f}'.format(shannon_diversity(all_giants['us-powergrid'])))

pp.table(cellText=table_data, rowLabels=row_labels, colLabels=col_labels,
         colWidths=[0.15]*4, cellLoc='center', loc='center')
pp.axis('tight')
pp.axis('off')
pp.show()
pp.clf()
print('done')


#
# local clustering distributions
#

print('local clustering distributions... ', end='', flush=True)

ax = pp.subplot(111)
bins = np.linspace(0, 1, 100)
ax.plot(bins, cumulative_cci_dist(all_giants['hamster'], bins), linewidth=2, label='hamster')
ax.plot(bins, cumulative_cci_dist(all_giants['euroroad'], bins), linewidth=2, label='euroroad')
ax.plot(bins, cumulative_cci_dist(all_giants['us-airports'], bins), linewidth=2, label='us-airports')
ax.plot(bins, cumulative_cci_dist(all_giants['us-powergrid'], bins), linewidth=2, label='us-powergrid')
ax.set_ylabel('Cumulative Probability', color=ppcolor1, alpha=0.8, fontsize=20)
ax.set_xlabel('Local Transitivity', color=ppcolor1, alpha=0.8, fontsize=20)
ax.legend(loc='lower right', prop={'size': 20})
stylize_plot(ax, False)
pp.show()
pp.clf()
print('done')


#
# shortest path distributions
#

print('shortest path distributions... ', end='', flush=True)

all_pds = {}
all_pds['hamster'] = shortest_path_dist(all_giants['hamster'])
all_pds['euroroad'] = shortest_path_dist(all_giants['euroroad'])
all_pds['us-airports'] = shortest_path_dist(all_giants['us-airports'])
all_pds['us-powergrid'] = shortest_path_dist(all_giants['us-powergrid'])

ax = pp.subplot(111)
ax.plot(all_pds['hamster'][1], lw=2.5, alpha=0.8, label='hamster')
ax.plot(all_pds['euroroad'][1], lw=2.5, alpha=0.8, label='euroroad')
ax.plot(all_pds['us-airports'][1], lw=2.5, alpha=0.8, label='us-airports')
ax.plot(all_pds['us-powergrid'][1], lw=2.5, alpha=0.8, label='us-powergrid')
ax.set_xlabel('Shortest Path Length', color=ppcolor1, alpha=0.8, fontsize=20)
ax.set_ylabel('Probability', color=ppcolor1, alpha=0.8, fontsize=20)
ax.legend(loc='upper right', prop={'size': 20})
stylize_plot(ax)
pp.show()
pp.clf()
print('done')


#
# centrality measurements
#

# first, calculate all distributions
print('centrality measurements', flush=True)

all_cents = {}
all_cents['hamster'] = {}
all_cents['euroroad'] = {}
all_cents['us-airports'] = {}
all_cents['us-powergrid'] = {}

# 1
all_cents['hamster']['closeness'] = all_giants['hamster'].closeness()
all_cents['hamster']['betweenness'] = all_giants['hamster'].betweenness(directed=False)
all_cents['hamster']['eigenvector'] = all_giants['hamster'].eigenvector_centrality(directed=False)
all_cents['hamster']['pagerank'] = all_giants['hamster'].pagerank(directed=False)

# 2
all_cents['euroroad']['closeness'] = all_giants['euroroad'].closeness()
all_cents['euroroad']['betweenness'] = all_giants['euroroad'].betweenness(directed=False)
all_cents['euroroad']['eigenvector'] = all_giants['euroroad'].eigenvector_centrality(directed=False)
all_cents['euroroad']['pagerank'] = all_giants['euroroad'].pagerank(directed=False)

# 3
all_cents['us-airports']['closeness'] = all_giants['us-airports'].closeness()
all_cents['us-airports']['betweenness'] = all_giants['us-airports'].betweenness(directed=False)
all_cents['us-airports']['eigenvector'] = all_giants['us-airports'].eigenvector_centrality(directed=False)
all_cents['us-airports']['pagerank'] = all_giants['us-airports'].pagerank(directed=False)

# 4
all_cents['us-powergrid']['closeness'] = all_giants['us-powergrid'].closeness()
all_cents['us-powergrid']['betweenness'] = all_giants['us-powergrid'].betweenness(directed=False)
all_cents['us-powergrid']['eigenvector'] = all_giants['us-powergrid'].eigenvector_centrality(directed=False)
all_cents['us-powergrid']['pagerank'] = all_giants['us-powergrid'].pagerank(directed=False)

# cross-correlate the distributions
print('└─ correlate all centralities', flush=True)

# 1
print('   ├─ hamster:', flush=True)
keys_only = list(all_cents['hamster'].keys())
vals_only = list(all_cents['hamster'].values())
for i in range(len(vals_only)):
    for j in range(i+1, len(vals_only)):
        print('   │  {:^11} x {:^11}: {: .4f}'.format(keys_only[i], keys_only[j], pearson_r(vals_only[i], vals_only[j])))

# 2
print('   ├─ euroroad:', flush=True)
keys_only = list(all_cents['euroroad'].keys())
vals_only = list(all_cents['euroroad'].values())
for i in range(len(vals_only)):
    for j in range(i+1, len(vals_only)):
        print('   │  {:^11} x {:^11}: {: .4f}'.format(keys_only[i], keys_only[j], pearson_r(vals_only[i], vals_only[j])))

# 3
print('   ├─ us-airports:', flush=True)
keys_only = list(all_cents['us-airports'].keys())
vals_only = list(all_cents['us-airports'].values())
for i in range(len(vals_only)):
    for j in range(i+1, len(vals_only)):
        print('   │  {:^11} x {:^11}: {: .4f}'.format(keys_only[i], keys_only[j], pearson_r(vals_only[i], vals_only[j])))

# 4
print('   └─ us-powergrid:', flush=True)
keys_only = list(all_cents['us-powergrid'].keys())
vals_only = list(all_cents['us-powergrid'].values())
for i in range(len(vals_only)):
    for j in range(i+1, len(vals_only)):
        print('      {:^11} x {:^11}: {: .4f}'.format(keys_only[i], keys_only[j], pearson_r(vals_only[i], vals_only[j])))

# normalize all data
bins = np.logspace(-2, 0, 20)
all_cents = {k1:
             {k2:
              list(map(lambda x: x/max(v2), v2))
              for k2, v2 in v1.items()}
             for k1, v1 in all_cents.items()}

# major ax
# fig = pp.figure()
# ax = pp.subplot(111)
# ax.spines['top'].set_color('none')
# ax.spines['left'].set_color('none')
# ax.spines['right'].set_color('none')
# ax.spines['bottom'].set_color('none')
# ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

# 1
ax1 = pp.subplot(221)
ax1.hist((all_cents['hamster']['closeness'],
          all_cents['hamster']['betweenness'],
          all_cents['hamster']['eigenvector'],
          all_cents['hamster']['pagerank']),
         bins, label=('closeness', 'betweenness', 'eigenvector', 'pagerank'))
ax1.set_xlabel('Normalized Centralities', color=ppcolor1, alpha=0.8, fontsize=20)
ax1.set_ylabel('Frequency', color=ppcolor1, alpha=0.8, fontsize=20)
ax1.set_title('Hamster', color=ppcolor1, alpha=0.8, fontsize=20)
ax1.legend(loc='upper center')
ax1.set_yscale('log')
ax1.set_xscale('log')
stylize_plot(ax1, False)

# 2
ax2 = pp.subplot(222)
ax2.hist((all_cents['euroroad']['closeness'],
          all_cents['euroroad']['betweenness'],
          all_cents['euroroad']['eigenvector'],
          all_cents['euroroad']['pagerank']),
         bins, label=('closeness', 'betweenness', 'eigenvector', 'pagerank'))
ax2.set_xlabel('Normalized Centralities', color=ppcolor1, alpha=0.8, fontsize=20)
ax2.set_ylabel('Frequency', color=ppcolor1, alpha=0.8, fontsize=20)
ax2.set_title('Euroroad', color=ppcolor1, alpha=0.8, fontsize=20)
ax2.legend(loc='upper center')
ax2.set_yscale('log')
ax2.set_xscale('log')
stylize_plot(ax2, False)

# 3
ax3 = pp.subplot(223)
ax3.hist((all_cents['us-airports']['closeness'],
          all_cents['us-airports']['betweenness'],
          all_cents['us-airports']['eigenvector'],
          all_cents['us-airports']['pagerank']),
         bins, label=('closeness', 'betweenness', 'eigenvector', 'pagerank'))
ax3.set_xlabel('Normalized Centralities', color=ppcolor1, alpha=0.8, fontsize=20)
ax3.set_ylabel('Frequency', color=ppcolor1, alpha=0.8, fontsize=20)
ax3.set_title('US Airports', color=ppcolor1, alpha=0.8, fontsize=20)
ax3.legend(loc='upper center')
ax3.set_yscale('log')
ax3.set_xscale('log')
stylize_plot(ax3, False)

# 4
ax4 = pp.subplot(224)
ax4.hist((all_cents['us-powergrid']['closeness'],
          all_cents['us-powergrid']['betweenness'],
          all_cents['us-powergrid']['eigenvector'],
          all_cents['us-powergrid']['pagerank']),
         bins, label=('closeness', 'betweenness', 'eigenvector', 'pagerank'))
ax4.set_xlabel('Normalized Centralities', color=ppcolor1, alpha=0.8, fontsize=20)
ax4.set_ylabel('Frequency', color=ppcolor1, alpha=0.8, fontsize=20)
ax4.set_title('US Powergrid', color=ppcolor1, alpha=0.8, fontsize=20)
ax4.legend(loc='upper left')
ax4.set_yscale('log')
ax4.set_xscale('log')
stylize_plot(ax4, False)

pp.tight_layout()
pp.show()
pp.clf()
print('done')

# TODO
# Scatter plot AxB greatest correlations for each network
# Evaluate for power-law fit the betweenness centrality distribution for each network
