import random_fields
import numpy as np
import scipy
import itertools
import hilbert_curve
import functools
import math
def generate_grid_coords(num_sensors, boundary):
    num_spaces = math.sqrt(num_sensors)
    assert (num_spaces-int(num_spaces)) == 0
    oned = np.linspace(start=0,stop=boundary,num=num_spaces)
    twod = list(itertools.product(*[oned,oned]))
    return twod
def generate_sorted_grid_coords(num_sensors, boundary_power_of_2=5):
    grid_size = 2**boundary_power_of_2-1
    coords = generate_grid_coords(num_sensors, grid_size)
    return sort_coords(coords, boundary_power_of_2)
def generate_sorted_network_coords(num_sensors, grid_power_of_2 = 5):
    grid_size = 2**grid_power_of_2-1
    network_coords = generate_network_coords(num_sensors, size = grid_size)
    return sort_coords(network_coords, grid_power_of_2)
def sort_coords(network_coords, grid_power_of_2=5):
    hilbert = functools.partial(hilbert_curve.point_to_hilbert,grid_power_of_2)
    #normalised = [(round(x), round(y)) for x, y in network_coords]
    inorder = sorted(network_coords, key = lambda i: hilbert(int(round(i[0])), int(round(i[1]))))
    return inorder
def generate_random_coords(size):
    return (np.random.uniform(0, size), np.random.uniform(0,size))
def generate_network_coords(num_sensors, size=32):
    return [generate_random_coords(size) for _ in range(num_sensors)]
def pairwise_distances(sensors):
    return scipy.spatial.distance.pdist(sensors)
def connection_probability(r_char,n, dist):
    beta = 1
    ponent = (dist/r_char)**n
    return beta* np.exp(-(ponent))
def coin_flip(p):
    return p>np.random.uniform()
def translate_vector_form(num_items, vector):
    actualidxs = list(itertools.combinations(range(num_items), 2))
    res = {actualidxs[i]:el for i,el in enumerate(vector)}
    return res
def soft_geometric_graph(sensors, r_char=32,n=1):
    connector = functools.partial(connection_probability, r_char, n)
    probabilities = connector(pairwise_distances(sensors))
    #connections = coin_flip(probabilities)
    connections = [coin_flip(p) for p in probabilities]
    readable = translate_vector_form(len(sensors), connections)
    return [i for i, val in readable.items() if val]
def get_adjacency_list(graph):
    inorder = sorted(graph, key = lambda tpl:tpl[0])
    eachone = itertools.groupby(inorder, key = lambda tpl:tpl[0])
    return {k:[i[1] for i in v]  for k,v in eachone}
def num_to_name(num):
    return 'S'+str(num)
def name_to_num(name):
    return int(name.strip('S'))
def is_connected_to_active(sensors, active_sensors, r_char=12,n=1):
    graph = soft_geometric_graph(sensors, r_char,n)
    adjacency_list = get_adjacency_list(graph)
    active_sensor_nums = [name_to_num(i) for i in active_sensors]
    adjacent_to_active = set(itertools.chain(*[v for k,v in adjacency_list.items() if k in active_sensor_nums]))
    adjacent_to_active |= set(active_sensor_nums)
    return {num_to_name(idx): (idx in adjacent_to_active) for idx,_ in enumerate(sensors)}

