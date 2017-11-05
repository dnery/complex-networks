import numpy as np

from os import path
from subprocess import call
from collections import namedtuple
from sklearn.metrics import normalized_mutual_info_score

from igraph import Graph
from igraph import VertexClustering

from matplotlib import pyplot as pp


#
# Pretty-print functions
#

def stylize_plot(ax, xlab, ylab, lloc='upper right', grid=True):
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
    ax.set_xlabel(xlab, color=ppcolor1, alpha=0.8, fontsize=20)
    ax.set_ylabel(ylab, color=ppcolor1, alpha=0.8, fontsize=20)
    ax.legend(loc=lloc, prop={'size': 20})


#
# Actually useful functions
#

def pearson_r(a, b):
    na = len(a)
    nb = len(b)
    if na != nb:
        raise ValueError('Samples differ in length')
    # standard sum
    sum_a = sum([float(x) for x in a])
    sum_b = sum([float(x) for x in b])
    # product sum
    sum_ab = sum([x * knn_k for x, knn_k in zip(a, b)])
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


def fastgreedy_progression(graph):
    optimal_structure = graph.community_fastgreedy()
    optimal_count = optimal_structure.optimal_count
    x_axis = range(1, optimal_count + 1)
    y_axis = np.zeros(len(x_axis))
    for i in x_axis:
        y_axis[i - 1] = optimal_structure.as_clustering(n=i).modularity
    return x_axis, y_axis, optimal_count


def retrieve_graph_and_clustering(benchmark_executable_path, mixing_param):
    call([str(benchmark_executable_path), '-N', '250', '-k', '7', '-maxk', '30', '-mu', str(mixing_param)])
    with open('network.dat', 'rb') as g_file:
        graph = Graph.Read_Edgelist(g_file, directed=False)
    with open('community.dat', 'rb') as c_file:
        memberships = [int(line.split()[1]) for line in c_file]
    graph.delete_vertices(0)
    graph.simplify()
    return graph, memberships

#
# global configs
#

ppcolor1 = '#000088'
ppcolor2 = '#ccffcc'


#
# Actual script
#

# 1
g_names = ['corticalCat',
           'corticalHuman',
           'corticalMonkey',
           'usAirports',
           'euroroad',
           'hamster']

g_giants = []
for g_name in g_names:
    g_path = path.join('networks', g_name + '.txt')
    with open(g_path, 'rb') as g_file:
        giant = Graph.Read_Edgelist(g_file, directed=False)
        giant = giant.components().giant()
        giant.simplify()
        g_giants.append(giant)

print('\nAssortativity')
print('=============\n')
print(g_names[0], '= {:.4f}'.format(g_giants[0].assortativity_degree()))
print(g_names[1], '= {:.4f}'.format(g_giants[1].assortativity_degree()))
print(g_names[2], '= {:.4f}'.format(g_giants[2].assortativity_degree()))
print(g_names[3], '= {:.4f}'.format(g_giants[3].assortativity_degree()))
print(g_names[4], '= {:.4f}'.format(g_giants[4].assortativity_degree()))
print(g_names[5], '= {:.4f}'.format(g_giants[5].assortativity_degree()))
print('done.')

# 2
knn_x = []
ax = pp.subplot(111)
for i_giant in range(len(g_giants)):
    knn_tuple = g_giants[i_giant].knn()
    knn_x.append(knn_tuple[0])
    knn_k = knn_tuple[1]
    k = range(1, len(knn_k) + 1)
    ax.plot(k, knn_k, ls='', marker='o', label=g_names[i_giant])
stylize_plot(ax, 'Degree k', 'knn(k)')
pp.show()
pp.clf()

# 3
print('\nCorrelations (k(x) with knn(x))')
print('===============================\n')
print(g_names[0], '= {:.4f}'.format(pearson_r(g_giants[0].degree(), knn_x[0])))
print(g_names[1], '= {:.4f}'.format(pearson_r(g_giants[1].degree(), knn_x[1])))
print(g_names[2], '= {:.4f}'.format(pearson_r(g_giants[2].degree(), knn_x[2])))
print(g_names[3], '= {:.4f}'.format(pearson_r(g_giants[3].degree(), knn_x[3])))
print(g_names[4], '= {:.4f}'.format(pearson_r(g_giants[4].degree(), knn_x[4])))
print(g_names[5], '= {:.4f}'.format(pearson_r(g_giants[5].degree(), knn_x[5])))
print('done.')

# 4
# if(False):
print('\nBest community structure modularities')
print('=====================================\n')

print('\nEdge betweenness...', flush=True)
print(g_names[0] + ' = ' + '{:.4f}'.format(g_giants[0].community_edge_betweenness(directed=False).as_clustering().modularity))
print(g_names[1] + ' = ' + '{:.4f}'.format(g_giants[1].community_edge_betweenness(directed=False).as_clustering().modularity))
print(g_names[2] + ' = ' + '{:.4f}'.format(g_giants[2].community_edge_betweenness(directed=False).as_clustering().modularity))
# print(g_names[3] + ' = ' + '{:.4f}'.format(g_giants[3].community_edge_betweenness(directed=False).as_clustering().modularity))
print(g_names[4] + ' = ' + '{:.4f}'.format(g_giants[4].community_edge_betweenness(directed=False).as_clustering().modularity))
# print(g_names[5] + ' = ' + '{:.4f}'.format(g_giants[5].community_edge_betweenness(directed=False).as_clustering().modularity))

print('\nFastGreedy algorithm...', flush=True)
print(g_names[0] + ' = ' + '{:.4f}'.format(g_giants[0].community_fastgreedy().as_clustering().modularity))
print(g_names[1] + ' = ' + '{:.4f}'.format(g_giants[1].community_fastgreedy().as_clustering().modularity))
print(g_names[2] + ' = ' + '{:.4f}'.format(g_giants[2].community_fastgreedy().as_clustering().modularity))
print(g_names[3] + ' = ' + '{:.4f}'.format(g_giants[3].community_fastgreedy().as_clustering().modularity))
print(g_names[4] + ' = ' + '{:.4f}'.format(g_giants[4].community_fastgreedy().as_clustering().modularity))
print(g_names[5] + ' = ' + '{:.4f}'.format(g_giants[5].community_fastgreedy().as_clustering().modularity))

print('\nLeading eigenvectors...', flush=True)
print(g_names[0] + ' = ' + '{:.4f}'.format(g_giants[0].community_leading_eigenvector().modularity))
print(g_names[1] + ' = ' + '{:.4f}'.format(g_giants[1].community_leading_eigenvector().modularity))
print(g_names[2] + ' = ' + '{:.4f}'.format(g_giants[2].community_leading_eigenvector().modularity))
print(g_names[3] + ' = ' + '{:.4f}'.format(g_giants[3].community_leading_eigenvector().modularity))
print(g_names[4] + ' = ' + '{:.4f}'.format(g_giants[4].community_leading_eigenvector().modularity))
print(g_names[5] + ' = ' + '{:.4f}'.format(g_giants[5].community_leading_eigenvector().modularity))

print('\nWalktrap...', flush=True)
print(g_names[0] + ' = ' + '{:.4f}'.format(g_giants[0].community_walktrap(steps=10).as_clustering().modularity))
print(g_names[1] + ' = ' + '{:.4f}'.format(g_giants[1].community_walktrap(steps=10).as_clustering().modularity))
print(g_names[2] + ' = ' + '{:.4f}'.format(g_giants[2].community_walktrap(steps=10).as_clustering().modularity))
print(g_names[3] + ' = ' + '{:.4f}'.format(g_giants[3].community_walktrap(steps=10).as_clustering().modularity))
print(g_names[4] + ' = ' + '{:.4f}'.format(g_giants[4].community_walktrap(steps=10).as_clustering().modularity))
print(g_names[5] + ' = ' + '{:.4f}'.format(g_giants[5].community_walktrap(steps=10).as_clustering().modularity))
print('done.')

# 5
print('\nFast-greedy steps distribution')
print('==============================\n')

ax = pp.subplot(111)

print('\nSteps for usAirports network...')
x, y, optimal = fastgreedy_progression(g_giants[3])
print('  optimal count is', optimal)
ax.plot(x, y, label='usAirports')

print('\nSteps for euroroad network...')
x, y, optimal = fastgreedy_progression(g_giants[4])
print('  optimal count is', optimal)
ax.plot(x, y, label='euroroad')

stylize_plot(ax, 'Cluster Count', 'Modularity', lloc='lower right')
print('done.')
pp.show()
pp.clf()

# 6
print('\nCommunity finding algorithms comparison')
print('=======================================\n')

x_axis = np.linspace(0.1, 1.0, 10)
nmi_leading_eigenvector = []
nmi_edge_betweenness = []
nmi_fastgreedy = []
nmi_walktrap = []
for param in x_axis:
    # run external program
    call(['./binary_networks/benchmark', '-N', '250', '-k', '7', '-maxk', '30', '-mu', str(param)])
    # retrieve graph and memberships
    with open('network.dat', 'rb') as g_file:
        graph = Graph.Read_Edgelist(g_file, directed=False)
    with open('community.dat', 'rb') as c_file:
        memberships = [int(line.split()[1]) for line in c_file]
    graph.delete_vertices(0)
    graph.simplify()
    # retrieve real and calculated modularities
    nmi_fastgreedy.append(normalized_mutual_info_score(memberships, graph.community_fastgreedy().as_clustering().membership))
    nmi_leading_eigenvector.append(normalized_mutual_info_score(memberships, graph.community_leading_eigenvector().membership))
    nmi_edge_betweenness.append(normalized_mutual_info_score(memberships, graph.community_edge_betweenness(directed=False).as_clustering().membership))
    nmi_walktrap.append(normalized_mutual_info_score(memberships, graph.community_walktrap(steps=10).as_clustering().membership))

ax = pp.subplot(111)
ax.plot(x_axis, nmi_walktrap, label='walktrap')
ax.plot(x_axis, nmi_fastgreedy, label='fastgreedy')
ax.plot(x_axis, nmi_edge_betweenness, label='edgeBetweenness')
ax.plot(x_axis, nmi_leading_eigenvector, label='leadingEigenvector')
stylize_plot(ax, 'Mixing Parameter', 'NMI (Real Communities vs Algorithms)')
print('done.')
pp.show()
pp.clf()