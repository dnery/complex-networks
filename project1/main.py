import networkx as nx
import powerlaw as pl
import numpy as np

from os import path
from matplotlib import pyplot as pp


#
# function definitions
#

def moment(graph, which):
    measurement = 0
    for node in nx.nodes(graph):
        measurement += nx.degree(graph, node) ** which
    return measurement / nx.number_of_nodes(graph)


def shortest_path_dist(graph):
    # remember, ints from strings!!!
    nodes = list(map(int, nx.nodes(graph)))
    freqs = np.zeros(nx.diameter(graph)+1, dtype=int)
    mem_dims = max(nodes) + 1  # memorize repeated data
    memory = np.zeros((mem_dims, mem_dims), dtype=int) - 1
    for node1 in nodes:
        for node2 in nodes:
            if node1 == node2:
                continue
            if memory[node1, node2] > -1:
                freqs[memory[node1, node2]] += 1
            else:
                length = int(nx.shortest_path_length(graph, source=str(node1), target=str(node2)))
                memory[node1, node2] = length
                memory[node2, node1] = length
                freqs[length] += 1
    return freqs


#
# read giant components
#

print('=> reading networks...', end='')

all_graphs = {}
all_giants = {}

current_path = path.join('networks', 'hamster.txt')
with open(current_path, 'rb') as graph_file:
    graph = nx.read_edgelist(graph_file)
    all_graphs['hamster'] = graph
    all_giants['hamster'] = max(nx.connected_component_subgraphs(graph), key=len)

current_path = path.join('networks', 'euroroad.txt')
with open(current_path, 'rb') as graph_file:
    graph = nx.read_edgelist(graph_file)
    all_graphs['euroroad'] = graph
    all_giants['euroroad'] = max(nx.connected_component_subgraphs(graph), key=len)

current_path = path.join('networks', 'us-airports.txt')
with open(current_path, 'rb') as graph_file:
    graph = nx.read_weighted_edgelist(graph_file)
    all_graphs['us-airports'] = graph
    all_giants['us-airports'] = max(nx.connected_component_subgraphs(graph), key=len)

current_path = path.join('networks', 'us-powergrid.txt')
with open(current_path, 'rb') as graph_file:
    graph = nx.read_weighted_edgelist(graph_file)
    all_graphs['us-powergrid'] = graph
    all_giants['us-powergrid'] = max(nx.connected_component_subgraphs(graph), key=len)

print(' done')

#
# degree distributions
#

print('=> degree distributions, p-law alphas...')

all_pdfs = {}

n_nodes = nx.number_of_nodes(all_giants['hamster'])
all_pdfs['hamster'] = list(map(lambda x: x/n_nodes, nx.degree_histogram(all_giants['hamster'])))
n_nodes = nx.number_of_nodes(all_giants['euroroad'])
all_pdfs['euroroad'] = list(map(lambda x: x/n_nodes, nx.degree_histogram(all_giants['euroroad'])))
n_nodes = nx.number_of_nodes(all_giants['us-airports'])
all_pdfs['us-airports'] = list(map(lambda x: x/n_nodes, nx.degree_histogram(all_giants['us-airports'])))
n_nodes = nx.number_of_nodes(all_giants['us-powergrid'])
all_pdfs['us-powergrid'] = list(map(lambda x: x/n_nodes, nx.degree_histogram(all_giants['us-powergrid'])))

all_fits = {}

all_fits['hamster'] = pl.Fit(all_pdfs['hamster'])
print('  hamster:\n\talpha:{:.4f}'.format(all_fits['hamster'].alpha))
all_fits['euroroad'] = pl.Fit(all_pdfs['euroroad'])
print('  euroroad:\n\talpha:{:.4f}'.format(all_fits['euroroad'].alpha))
all_fits['us-airports'] = pl.Fit(all_pdfs['us-airports'])
print('  us-airports:\n\talpha:{:.4f}'.format(all_fits['us-airports'].alpha))
all_fits['us-powergrid'] = pl.Fit(all_pdfs['us-powergrid'])
print('  us-powergrid:\n\talpha:{:.4f}'.format(all_fits['us-powergrid'].alpha))

print('done')


#
# collect measures
#

'''
table_data = np.zeros((7, 4))
col_labels = ('Hamster', 'Euroroad', 'US-airports', 'US-powergrid')
row_labels = ('NumberOfNodes', '<k>', '<k²>', '<cci>', 'Transitivity', '<ShortestPaths>', 'Diameter')

table_data[0,0] = nx.number_of_nodes(all_giants['hamster'])
table_data[0,1] = nx.number_of_nodes(all_giants['euroroad'])
table_data[0,2] = nx.number_of_nodes(all_giants['us-airports'])
table_data[0,3] = nx.number_of_nodes(all_giants['us-powergrid'])

table_data[1,0] = moment(all_giants['hamster'], 1)
table_data[1,1] = moment(all_giants['euroroad'], 1)
table_data[1,2] = moment(all_giants['us-airports'], 1)
table_data[1,3] = moment(all_giants['us-powergrid'], 1)

table_data[2,0] = moment(all_giants['hamster'], 2)
table_data[2,1] = moment(all_giants['euroroad'], 2)
table_data[2,2] = moment(all_giants['us-airports'], 2)
table_data[2,3] = moment(all_giants['us-powergrid'], 2)

table_data[3,0] = nx.average_clustering(all_giants['hamster'])
table_data[3,1] = nx.average_clustering(all_giants['euroroad'])
table_data[3,2] = nx.average_clustering(all_giants['us-airports'])
table_data[3,3] = nx.average_clustering(all_giants['us-powergrid'])

table_data[4,0] = nx.transitivity(all_giants['hamster'])
table_data[4,1] = nx.transitivity(all_giants['euroroad'])
table_data[4,2] = nx.transitivity(all_giants['us-airports'])
table_data[4,3] = nx.transitivity(all_giants['us-powergrid'])

table_data[5,0] = nx.average_shortest_path_length(all_giants['hamster'])
table_data[5,1] = nx.average_shortest_path_length(all_giants['euroroad'])
table_data[5,2] = nx.average_shortest_path_length(all_giants['us-airports'])
table_data[5,3] = nx.average_shortest_path_length(all_giants['us-powergrid'])

table_data[6,0] = nx.diameter(all_giants['hamster'])
table_data[6,1] = nx.diameter(all_giants['euroroad'])
table_data[6,2] = nx.diameter(all_giants['us-airports'])
table_data[6,3] = nx.diameter(all_giants['us-powergrid'])

pp.table(cellText=table_data, rowLabels=row_labels, colLabels=col_labels, loc='center')
pp.axis('tight')
pp.axis('off')
pp.show()
pp.clf()
'''

'''
pp.title('degree probability distributions')
p1, = pp.loglog(all_pdfs['hamster'], 'rs', label='hamster')
p2, = pp.loglog(all_pdfs['euroroad'], 'go', label='euroroad')
p3, = pp.loglog(all_pdfs['us-airports'], 'bd', label='us-airports')
p4, = pp.loglog(all_pdfs['us-powergrid'], 'yp', label='us-powergrid')
#pp.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
pp.legend(handles=[p1, p2, p3, p4], loc=1)
pp.grid(True)
pp.show()
pp.clf()
'''

'''
pp.title('shortest path distributions')
p1, = pp.plot(shortest_path_dist(all_giants['hamster']), 'rs', label='hamster')
p2, = pp.plot(shortest_path_dist(all_giants['euroroad']), 'go', label='euroroad')
p3, = pp.plot(shortest_path_dist(all_giants['us-airports']), 'bd', label='us-airports')
p4, = pp.plot(shortest_path_dist(all_giants['us-powergrid']), 'yp', label='us-powergrid')
#pp.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
pp.legend(handles=[p1, p2, p3, p4], loc=1)
pp.grid(True)
pp.show()
pp.clf()
'''

print('=> shortest path distributions...')

path_dists = {}

print('  hamster: ', end='')
path_dists['hamster'] = shortest_path_dist(all_giants['hamster'])
print(path_dists['hamster'])
print('  euroroad: ', end='')
path_dists['euroroad'] = shortest_path_dist(all_giants['euroroad'])
print(path_dists['euroroad'])
print('  us-airports: ', end='')
path_dists['us-airports'] = shortest_path_dist(all_giants['us-airports'])
print(path_dists['us-airports'])
print('  us-powergrid: ', end='')
path_dists['us-powergrid'] = shortest_path_dist(all_giants['us-powergrid'])
print(path_dists['us-powergrid'])

print('done')
