import random_fields
import numpy as np
import scipy
import itertools
def generate_random_coords(size):
    return (np.random.uniform(0, size), np.random.uniform(0,size))
def generate_network_coords(num_sensors, size=29):
    return [generate_random_coords(size) for _ in range(num_sensors)]
def pairwise_distances(sensors):
    return scipy.spatial.distance.pdist(sensors)
def connection_probability(r_char, dist):
    beta = 1
    n =1
    ponent = (dist/r_char)**n
    return beta* np.exp(-(ponent))
def coin_flip(p):
    return p>0.5
def translate_vector_form(num_items, vector):
    actualidxs = list(itertools.combinations(range(num_items), 2))
    res = {actualidxs[i]:el for i,el in enumerate(vector)}
    return res
def soft_geometric_graph(sensors, r_char=15):
    connector = functools.partial(connection_probability, r_char)
    connections = coin_flip(connector(pairwise_distances(sensors)))
    readable = translate_vector_form(len(sensors), connections)
    return [i for i, val in readable.items() if val]
def get_adjacency_list(graph):
    inorder = sorted(graph, key = lambda tpl:tpl[0])
    eachone = itertools.groupby(inorder, key = lambda tpl:tpl[0])
    return {k:[i[1] for i in v]  for k,v in eachone}
def num_to_name(num):
    return 'S'+str(num)
def is_connected_to_active(sensors, active_sensors):
    graph = soft_geometric_graph(sensors, r_char=15)
    adjacency_list = get_adjacency_list(graph)
    adjacent_to_active = set(itertools.chain(*[v for k,v in adjacency_list.items() if k in active_sensors]))
    adjacent_to_active |= set(active_sensors)
    return {num_to_name(idx): (idx in adjacent_to_active) for idx,_ in enumerate(sensors)}

