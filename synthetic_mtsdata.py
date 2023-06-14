from email.headerregistry import Group
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import torch
import sys
import os
import random
seed = 310058
np.random.seed(seed)
random.seed(seed)
root_dir = r"./data/synthetic"
T_length = 1200
graph_size = 20
K_of_graph = 3
rewire_rate = 0.5
node_value = np.linspace(-1, 1, graph_size)

num_series = 100

series_cluster = 2

cycle = 100

transition_graph_core = [nx.adjacency_matrix(nx.connected_watts_strogatz_graph(
    graph_size, K_of_graph, rewire_rate)).todense()for _ in range(series_cluster)]

cluster_member = num_series//series_cluster
assert cluster_member*series_cluster == num_series
transition_mat_list = []

for idx in range(num_series):
    core_idx = idx//cluster_member
    core_adj = transition_graph_core[core_idx]
    new_adj = np.zeros((graph_size, graph_size))
    for n_r, row in enumerate(core_adj):
        zeros_num = (row == 0).sum()
        add_edges = np.random.binomial(1, 1/(5*graph_size), zeros_num)
        # print(add_edges)
        row[row == 0] = add_edges
        new_adj[n_r] = row
        new_adj[n_r][n_r] = 1
    transition_mat_list.append(new_adj/new_adj.sum(axis=1, keepdims=True))

transition_mat_dict = {"mat_{}".format(
    k): transition_mat_list[k] for k in range(num_series)}
f_name = "transition_size{}_cluster{}_cycle{}_numsr{}.npz".format(
    graph_size, series_cluster, cycle, num_series)
np.savez(os.path.join(root_dir, f_name), **transition_mat_dict)


def oneloop(element_list):
    pop_ele = element_list.pop()
    element_list.insert(0, pop_ele)
    return element_list


def decide_adj(transition_list, move_idx, adj_in_group):
    total_num = len(transition_list)
    grouped_adj = []
    n_group = total_num//adj_in_group
    assert n_group*adj_in_group == total_num
    for idx in range(n_group):
        group = [transition_list[k]
                 for k in range(idx*adj_in_group, (idx+1)*adj_in_group)]
        grouped_adj.append(group)
    if move_idx > 0:
        reorder_grouped_adj = oneloop(grouped_adj)
        for move in range(move_idx-1):
            reorder_grouped_adj = oneloop(reorder_grouped_adj)
    else:
        reorder_grouped_adj = grouped_adj
    new_transition_list = []
    for group in reorder_grouped_adj:
        new_transition_list = new_transition_list+group
    return new_transition_list


mts_data = np.zeros((num_series, T_length)).astype(int)

for ts in range(1, T_length-1):
    circle_idx = ts//cycle
    current_transition_list = decide_adj(
        transition_mat_list, circle_idx, cluster_member)
    if ts % cycle == cycle-1 and ts > 0:
        current_states = mts_data[:, ts]
        bins = np.bincount(current_states)
        popular = np.argmax(bins)
        unique_v = np.sort(np.unique(current_states))
        popular_state = unique_v[popular]
        mts_data[:, ts+1] = popular_state
    else:
        for series_idx in range(num_series):
            latest_state = mts_data[series_idx][ts]
            transition_prob = current_transition_list[series_idx][int(
                latest_state)]
            randomwalk = np.random.choice(
                a=graph_size, size=1, p=transition_prob)[0]
            mts_data[series_idx][ts+1] = randomwalk
print(mts_data)
mts_data = mts_data.astype(float)
final_data = 2*(mts_data/graph_size)-1
f_name = "rwongraph_size{}_cluster{}_cycle{}_numsr{}.npy".format(
    graph_size, series_cluster, cycle, num_series)
np.save(os.path.join(root_dir, f_name), final_data)
