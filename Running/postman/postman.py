import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from collections import Counter
import networkx as nx

# node id: [(node id, distance, edge id)]
connections = {
	0: [(1, 0.29, 0), (25, 0.18, 24)],
	1: [(0, 0.29, 0), (2, 0.03, 1), (28, 0.06, 28)],
	2: [(1, 0.03, 1), (3, 0.08, 2), (32, 0.15, 29)],
	3: [(2, 0.08, 2), (4, 0.06, 3), (33, 0.13, 30)],
	4: [(3, 0.06, 3), (5, 0.07, 4), (35, 0.12, 31)],
	5: [(4, 0.07, 4), (6, 0.06, 5), (36, 0.12, 32)],
	6: [(5, 0.06, 5), (7, 0.07, 6), (37, 0.15, 33)],
	7: [(6, 0.07, 6), (8, 0.06, 7), (39, 0.15, 34)],
	8: [(7, 0.06, 7), (9, 0.06, 8), (40, 0.14, 35)],
	9: [(8, 0.06, 8), (10, 0.07, 9), (41, 0.14, 36)],
	10: [(9, 0.07, 9), (11, 0.03, 10), (43, 0.10, 37)],
	11: [(10, 0.03, 10), (12, 0.08, 11), (44, 0.07, 38)],
	12: [(11, 0.08, 11), (13, 0.10, 12), (44, 0.08, 39)],
	13: [(12, 0.10, 12), (14, 0.17, 13), (44, 0.08, 40)],
	14: [(13, 0.17, 13), (15, 0.08, 14), (45, 0.06, 41)],
	15: [(14, 0.08, 14), (16, 0.07, 15)],
	16: [(15, 0.07, 15), (17, 0.06, 16), (45, 0.05, 42)],
	17: [(16, 0.06, 16), (18, 0.07, 17), (46, 0.06, 44)],
	18: [(17, 0.07, 17), (19, 0.06, 18), (47, 0.08, 46)],
	19: [(18, 0.06, 18), (20, 0.03, 19), (48, 0.06, 48)],
	20: [(19, 0.03, 19), (22, 0.22, 20), (49, 0.07, 50)],
	# 21: [],
	22: [(20, 0.22, 20), (23, 0.20, 21), (51, 0.07, 53)],
	23: [(22, 0.20, 21), (24, 0.58, 22), (52, 0.07, 55)],
	24: [(23, 0.58, 22), (25, 0.14, 23), (29, 0.06, 57)],
	25: [(0, 0.18, 24), (24, 0.14, 23), (26, 0.12, 25)],
	26: [(25, 0.12, 25), (27, 0.07, 26), (29, 0.18, 58)],
	27: [(26, 0.07, 26), (28, 0.06, 27), (30, 0.16, 60)],
	28: [(1, 0.06, 28), (27, 0.06, 27), (31, 0.15, 62)],
	29: [(24, 0.06, 57), (26, 0.18, 58), (30, 0.06, 59), (52, 0.31, 56)],
	30: [(27, 0.16, 60), (29, 0.06, 59), (31, 0.06, 61)],
	31: [(28, 0.15, 62), (30, 0.06, 61), (32, 0.07, 63), (53, 0.22, 96)],
	32: [(2, 0.15, 29), (31, 0.07, 63), (33, 0.02, 64), (34, 0.03, 65)],
	33: [(3, 0.13, 30), (32, 0.02, 64), (34, 0.05, 66), (35, 0.07, 67)],
	34: [(32, 0.03, 65), (33, 0.05, 66), (61, 0.08, 68)],
	35: [(4, 0.15, 31), (33, 0.07, 67), (36, 0.07, 70), (62, 0.11, 69)],
	36: [(5, 0.12, 32), (35, 0.07, 70), (37, 0.07, 72), (57, 0.13, 71)],
	37: [(6, 0.15, 33), (36, 0.07, 72), (38, 0.03, 73)],
	38: [(37, 0.03, 73), (39, 0.05, 74), (56, 0.06, 84)],
	39: [(7, 0.15, 34), (38, 0.05, 74), (40, 0.07, 75), (49, 0.16, 83)],
	40: [(8, 0.14, 35), (39, 0.07, 75), (41, 0.06, 76), (48, 0.16, 82)],
	41: [(9, 0.14, 36), (40, 0.06, 76), (42, 0.06, 77), (47, 0.15, 81)],
	42: [(41, 0.06, 77), (43, 0.05, 78), (46, 0.16, 80)],
	43: [(10, 0.10, 37), (42, 0.05, 78), (44, 0.04, 102), (45, 0.20, 79)],
	44: [(11, 0.07, 38), (12, 0.08, 39), (13, 0.08, 40), (43, 0.04, 102)],
	45: [(14, 0.06, 41), (16, 0.05, 42), (43, 0.20, 79), (46, 0.06, 43)],
	46: [(17, 0.06, 44), (42, 0.16, 80), (45, 0.06, 43), (47, 0.07, 45)],
	47: [(18, 0.08, 46), (41, 0.15, 81), (46, 0.07, 45), (48, 0.07, 47)],
	48: [(19, 0.06, 48), (40, 0.16, 82), (47, 0.07, 47), (49, 0.06, 49)],
	49: [(20, 0.07, 50), (39, 0.16, 83), (48, 0.06, 49), (50, 0.05, 51)],
	50: [(49, 0.05, 51), (51, 0.17, 52), (55, 0.05, 86)],
	51: [(22, 0.07, 53), (50, 0.17, 52), (52, 0.16, 54), (54, 0.07, 91)],
	52: [(23, 0.07, 55), (29, 0.31, 56), (51, 0.16, 54), (53, 0.08, 94)],
	53: [(31, 0.22, 96), (52, 0.08, 94), (54, 0.14, 93), (60, 0.11, 95)],
	54: [(51, 0.07, 91), (53, 0.14, 93), (57, 0.13, 90), (59, 0.11, 92)],
	55: [(50, 0.05, 86), (56, 0.05, 85), (57, 0.04, 88)],
	56: [(38, 0.06, 84), (55, 0.05, 85), (57, 0.03, 87)],
	57: [(36, 0.13, 71), (54, 0.13, 90), (55, 0.04, 88), (56, 0.03, 87), (58, 0.04, 89)],
	58: [(57, 0.04, 89), (59, 0.02, 100), (62, 0.02, 99)],
	59: [(54, 0.11, 92), (58, 0.02, 100), (60, 0.03, 101)],
	60: [(53, 0.11, 95), (59, 0.03, 101), (61, 0.02, 97)],
	61: [(34, 0.08, 68), (60, 0.02, 97), (62, 0.03, 98)],
	62: [(35, 0.11, 69), (58, 0.02, 99), (61, 0.03, 98)]
	}

edges_list = []
for i in connections:
	for j in connections[i]:
		edges_list.append(j[2])

edges_list = list(set(edges_list))
edges = len(edges_list)

def convert_to_graph(connections_dict):

	edges = []
	for i in connections_dict:
		for j in connections_dict[i]:
			edge_ij = (i, j[0], {"weight": j[1]})
			edges.append(edge_ij)

	G = nx.Graph()
	G.add_edges_from(edges)

	return G, edges

G, e = convert_to_graph(connections_dict = connections)

def get_nodes_by_edge(connections_dict, edges_list):

	nodes_by_edge = {i: [] for i in edges_list}

	for i in connections_dict:
		for j in connections_dict[i]:
			nodes_by_edge[j[2]].append(i)

	return nodes_by_edge

nodes_by_edge = get_nodes_by_edge(
	connections_dict = connections,
	edges_list = edges_list
	)

def calc_min_dist(connections_dict):

	connections_list = list(connections_dict.values())
	connections_flat = [i for j in connections_list for i in j]
	distances = [i[1] for i in connections_flat]

	return sum(distances) / 2

optimal_distance = calc_min_dist(connections)

def weight_distances(connections_dict):

	loc_weights = {i: None for i in connections_dict}
	for i in connections_dict:
		options_dist = [j[1] for j in connections_dict[i]]
		options_dist_weighted = [sum(options_dist) / j for j in options_dist]
		options_weights_norm = [j / sum(options_dist_weighted) for j in options_dist_weighted]
		loc_weights[i] = options_weights_norm

	return loc_weights

loc_weights = weight_distances(connections_dict = connections)

def calc_metrics(optimal_distance, actual_distance, edge_path):

	route_eff = optimal_distance / actual_distance
	extra_distance = actual_distance - optimal_distance

	edges_explored_n = Counter(edge_path)
	most_explored_edge = max(edges_explored_n, key = edges_explored_n.get)
	most_explored_n = edges_explored_n[most_explored_edge]

	return route_eff, extra_distance, most_explored_edge, most_explored_n

def find_best_iteration(distance_cum):

	best_i = pd.Series(distance_cum).idxmin()

	return best_i

def simulate(f, n, *args, **kwargs):
	a = []
	b = []
	c = []
	d = []
	for _ in range(n):
		a_i, b_i, c_i, d_i = f(*args, **kwargs)
		a.append(a_i)
		b.append(b_i)
		c.append(c_i)
		d.append(d_i)

	return a, b, c, d

def prune_path(node_path, edge_path, distance_path):

	remove_i = []

	for i in range(len(edge_path)-2):
		if (edge_path[i] == edge_path[i + 1]) & (edge_path[i] == edge_path[i + 2]):
			remove_i.extend([i+1, i+2])

	node_path = [i for j, i in enumerate(node_path) if j not in remove_i]
	edge_path = [i for j, i in enumerate(edge_path) if j not in remove_i]
	distance_path = [i for j, i in enumerate(distance_path) if j not in remove_i]

	distance_cum = sum(distance_path)

	return distance_cum, node_path, edge_path, distance_path

n = 100

# Example - Model 1: Random ##################################################

def postman_random(connections_dict, edges_n, start = 0):

	current_loc = start
	distance_cum = 0
	node_path = [current_loc]
	edge_path = []
	distance_path = []

	while len(set(edge_path)) != edges_n:

		path = random.choice(connections_dict[current_loc])
		distance_cum += path[1]
		current_loc = path[0]
		node_path.append(current_loc)
		edge_path.append(path[2])
		distance_path.append(path[1])

	return distance_cum, node_path, edge_path, distance_path

distance_cum_i1_unpruned, node_path_i1_unpruned, edge_path_i1_unpruned, distance_path_i1_unpruned = postman_random(
	connections_dict = connections,
	edges_n = edges,
	start = 0
	)

route_eff_i1_unpruned, extra_distance_i1_unpruned, *others = calc_metrics(
	optimal_distance = optimal_distance,
	actual_distance = distance_cum_i1_unpruned,
	edge_path = edge_path_i1_unpruned
	)

# print("Model 1: Random -----------------------------------------------------")
# print("Distance:", round(distance_cum_i1_unpruned, 2))
# print("Route Efficiency:", round(route_eff_i1_unpruned, 2))
# print("Extra Distance:", round(extra_distance_i1_unpruned, 2))
# print(node_path_i1_unpruned)

## Simulation and prune example ##############################################

distance_cum_i1, node_path_i1, edge_path_i1, distance_path_i1 = prune_path(
	node_path = node_path_i1_unpruned,
	edge_path = edge_path_i1_unpruned,
	distance_path = distance_path_i1_unpruned
	)

# print(distance_cum_i1)
# print(node_path_i1)

# Model 1: Random ############################################################

distance_cum_m1_full, node_path_m1_full, edge_path_m1_full, distance_path_m1_full = simulate(
	f = postman_random,
	n = n,
	connections_dict = connections,
	edges_n = edges,
	start = 0
	)

distance_cum_m1 = []
node_path_m1 = []
edge_path_m1 = []
distance_path_m1 = []
for i in range(len(distance_cum_m1_full)):

	distance_cum_m1_i, node_path_m1_i, edge_path_m1_i, distance_path_m1_i = prune_path(
		node_path = node_path_m1_full[i],
		edge_path = edge_path_m1_full[i],
		distance_path = distance_path_m1_full[i]
		)
	distance_cum_m1.append(distance_cum_m1_i)
	node_path_m1.append(node_path_m1_i)
	edge_path_m1.append(edge_path_m1_i)
	distance_path_m1.append(distance_path_m1_i)

i_m1 = find_best_iteration(distance_cum_m1)

distance_cum_m1_best = distance_cum_m1[i_m1]
node_path_m1_best = node_path_m1[i_m1]
edge_path_m1_best = edge_path_m1[i_m1]
distance_path_m1_best = distance_path_m1[i_m1]

route_eff_m1, extra_distance_m1, most_explored_edge_m1, most_explored_n_m1 = calc_metrics(
	optimal_distance = optimal_distance,
	actual_distance = distance_cum_m1_best,
	edge_path = edge_path_m1_best
	)

progress_m1 = []
for i in range(len(edge_path_m1_best)):
	progress_m1.append(len(set(edge_path_m1_best[:i])) / edges)

fig_m1, ax_m1 = plt.subplots(1, 1)
ax_m1.hist(distance_cum_m1)
ax_m1.set_xlabel("Distance Travelled")
ax_m1.set_ylabel("Count")
fig_m1.suptitle("Random Path Generation", fontsize = 16)
plt.title("Average Distance = " + str(round(sum(distance_cum_m1) / n, 1)) + " miles", loc = "left")
plt.savefig("./viz/m1_dist_hist.png")
plt.close()

progress_m1_df = pd.DataFrame({
	"distance": np.cumsum(np.array(distance_path_m1_best)),
	"pct_edges_complete": progress_m1})
fig_m1_progress, ax_m1_progress = plt.subplots(1, 1)
ax_m1_progress.plot(progress_m1_df["distance"], progress_m1_df["pct_edges_complete"])
ax_m1_progress.set_xlabel("Distance Travelled")
ax_m1_progress.set_ylabel("Proportion of Edges Traversed")
fig_m1_progress.suptitle("Random Path Generation", fontsize = 16)
plt.savefig("./viz/m1_progress.png")
plt.close()

print("--- Model 1: Random -------------------------------------------------")
print("Distance:", round(distance_cum_m1_best, 2))
print("Route Efficiency:", round(route_eff_m1, 2))

# Model 2: Distance-Weighted #################################################

def postman_distance_weighted(connections_dict, edges_n, weights, start = 0):

	current_loc = start
	distance_cum = 0
	node_path = [current_loc]
	edge_path = []
	distance_path = []

	while len(set(edge_path)) != edges_n:

		path = connections_dict[current_loc][np.random.choice(
			len(connections_dict[current_loc]),
			p = weights[current_loc]
			)]
		distance_cum += path[1]
		current_loc = path[0]
		node_path.append(current_loc)
		edge_path.append(path[2])
		distance_path.append(path[1])

	return distance_cum, node_path, edge_path, distance_path

distance_cum_m2_full, node_path_m2_full, edge_path_m2_full, distance_path_m2_full = simulate(
	f = postman_distance_weighted,
	n = n,
	connections_dict = connections,
	edges_n = edges,
	weights = loc_weights,
	start = 0
	)

distance_cum_m2 = []
node_path_m2 = []
edge_path_m2 = []
distance_path_m2 = []
for i in range(len(distance_cum_m2_full)):

	distance_cum_m2_i, node_path_m2_i, edge_path_m2_i, distance_path_m2_i = prune_path(
		node_path = node_path_m2_full[i],
		edge_path = edge_path_m2_full[i],
		distance_path = distance_path_m2_full[i]
		)
	distance_cum_m2.append(distance_cum_m2_i)
	node_path_m2.append(node_path_m2_i)
	edge_path_m2.append(edge_path_m2_i)
	distance_path_m2.append(distance_path_m2_i)

i_m2 = find_best_iteration(distance_cum_m2)

distance_cum_m2_best = distance_cum_m2[i_m2]
node_path_m2_best = node_path_m2[i_m2]
edge_path_m2_best = edge_path_m2[i_m2]
distance_path_m2_best = distance_path_m2[i_m2]

route_eff_m2, extra_distance_m2, most_explored_edge_m2, most_explored_n_m2 = calc_metrics(
	optimal_distance = optimal_distance,
	actual_distance = distance_cum_m2_best,
	edge_path = edge_path_m2_best
	)

progress_m2 = []
for i in range(len(edge_path_m2_best)):
	progress_m2.append(len(set(edge_path_m2_best[:i])) / edges)

fig_m2, ax_m2 = plt.subplots(1, 1)
ax_m2.hist(distance_cum_m2)
ax_m2.set_xlabel("Distance Travelled")
ax_m2.set_ylabel("Count")
fig_m2.suptitle("Distance-Weighted Random Path Generation", fontsize = 16)
plt.title("Average Distance = " + str(round(sum(distance_cum_m2) / n, 1)) + " miles", loc = "left")
plt.savefig("./viz/m2_dist_hist.png")
plt.close()

progress_m2_df = pd.DataFrame({
	"distance": np.cumsum(np.array(distance_path_m2_best)),
	"pct_edges_complete": progress_m2})
fig_m2_progress, ax_m2_progress = plt.subplots(1, 1)
ax_m2_progress.plot(progress_m2_df["distance"], progress_m2_df["pct_edges_complete"])
ax_m2_progress.set_xlabel("Distance Travelled")
ax_m2_progress.set_ylabel("Proportion of Edges Traversed")
fig_m2_progress.suptitle("Distance-Weighted Random Path Generation", fontsize = 16)
plt.savefig("./viz/m2_progress.png")
plt.close()

print("--- Model 2: Distance-Weighted --------------------------------------")
print("Distance:", round(distance_cum_m2_best, 2))
print("Route Efficiency:", round(route_eff_m2, 2))

# Model 3: Exploration-Weighted ##############################################

def postman_exploration_weighted(connections_dict, edges_n, start = 0, prob_explored = 0.1):

	current_loc = start
	distance_cum = 0
	node_path = [current_loc]
	edge_path = []
	distance_path = []

	while len(set(edge_path)) != edges_n:
		
		node_options = [i[0] for i in connections_dict[current_loc]]
		len_options = len(node_options)
		len_unexplored_options = len([i for i in node_options if i not in node_path])
		len_explored_options = len_options - len_unexplored_options

		if (len_unexplored_options != 0) & (len_unexplored_options != len_options):
			prob_unexplored = (1 - (len_explored_options * prob_explored)) / len_unexplored_options
			probs_exploration = []
			for i in node_options:
				if i in node_path:
					probs_exploration.append(prob_explored)
				else:
					probs_exploration.append(prob_unexplored)

			path = connections_dict[current_loc][np.random.choice(
				len(connections_dict[current_loc]),
				p = probs_exploration
				)]

		else:

			path = connections_dict[current_loc][np.random.choice(
				len(connections_dict[current_loc])
				)]
		
		distance_cum += path[1]
		current_loc = path[0]
		node_path.append(current_loc)
		edge_path.append(path[2])
		distance_path.append(path[1])

	return distance_cum, node_path, edge_path, distance_path

distance_cum_m3_full, node_path_m3_full, edge_path_m3_full, distance_path_m3_full = simulate(
	f = postman_exploration_weighted,
	n = n,
	connections_dict = connections,
	edges_n = edges,
	prob_explored = 0.1,
	start = 0
	)

distance_cum_m3 = []
node_path_m3 = []
edge_path_m3 = []
distance_path_m3 = []
for i in range(len(distance_cum_m3_full)):

	distance_cum_m3_i, node_path_m3_i, edge_path_m3_i, distance_path_m3_i = prune_path(
		node_path = node_path_m3_full[i],
		edge_path = edge_path_m3_full[i],
		distance_path = distance_path_m3_full[i]
		)
	distance_cum_m3.append(distance_cum_m3_i)
	node_path_m3.append(node_path_m3_i)
	edge_path_m3.append(edge_path_m3_i)
	distance_path_m3.append(distance_path_m3_i)

i_m3 = find_best_iteration(distance_cum_m3)

distance_cum_m3_best = distance_cum_m3[i_m3]
node_path_m3_best = node_path_m3[i_m3]
edge_path_m3_best = edge_path_m3[i_m3]
distance_path_m3_best = distance_path_m3[i_m3]

route_eff_m3, extra_distance_m3, most_explored_edge_m3, most_explored_n_m3 = calc_metrics(
	optimal_distance = optimal_distance,
	actual_distance = distance_cum_m3_best,
	edge_path = edge_path_m3_best
	)

progress_m3 = []
for i in range(len(edge_path_m3_best)):
	progress_m3.append(len(set(edge_path_m3_best[:i])) / edges)

fig_m3, ax_m3 = plt.subplots(1, 1)
ax_m3.hist(distance_cum_m3)
ax_m3.set_xlabel("Distance Travelled")
ax_m3.set_ylabel("Count")
fig_m3.suptitle("Exploration Status-Weighted Random Path Generation", fontsize = 16)
plt.title("Average Distance = " + str(round(sum(distance_cum_m3) / n, 1)) + " miles", loc = "left")
plt.savefig("./viz/m3_dist_hist.png")
plt.close()

progress_m3_df = pd.DataFrame({
	"distance": np.cumsum(np.array(distance_path_m3_best)),
	"pct_edges_complete": progress_m3})
fig_m3_progress, ax_m3_progress = plt.subplots(1, 1)
ax_m3_progress.plot(progress_m3_df["distance"], progress_m3_df["pct_edges_complete"])
ax_m3_progress.set_xlabel("Distance Travelled")
ax_m3_progress.set_ylabel("Proportion of Edges Traversed")
fig_m3_progress.suptitle("Exploration Status-Weighted Random Path Generation", fontsize = 16)
plt.savefig("./viz/m3_progress.png")
plt.close()

print("--- Model 3: Exploration Status-Weighted ----------------------------")
print("Distance:", round(distance_cum_m3_best, 2))
print("Route Efficiency:", round(route_eff_m3, 2))

# Model 4: Random, No Backtracking ###########################################

def postman_random_forward(connections_dict, edges_n, start = 0):

	current_loc = start
	distance_cum = 0
	node_path = [current_loc]
	edge_path = []
	distance_path = []

	while len(set(edge_path)) != edges_n:

		if len(edge_path) != 0:
			prev_node = node_path[-2]
			options = [i for i in connections_dict[current_loc] if i[0] != prev_node]
			path = random.choice(options)
		else:
			path = random.choice(connections_dict[current_loc])

		distance_cum += path[1]
		current_loc = path[0]
		node_path.append(current_loc)
		edge_path.append(path[2])
		distance_path.append(path[1])

	return distance_cum, node_path, edge_path, distance_path

distance_cum_m4_full, node_path_m4_full, edge_path_m4_full, distance_path_m4_full = simulate(
	f = postman_exploration_weighted,
	n = n,
	connections_dict = connections,
	edges_n = edges,
	start = 0
	)

distance_cum_m4 = []
node_path_m4 = []
edge_path_m4 = []
distance_path_m4 = []
for i in range(len(distance_cum_m4_full)):

	distance_cum_m4_i, node_path_m4_i, edge_path_m4_i, distance_path_m4_i = prune_path(
		node_path = node_path_m4_full[i],
		edge_path = edge_path_m4_full[i],
		distance_path = distance_path_m4_full[i]
		)
	distance_cum_m4.append(distance_cum_m4_i)
	node_path_m4.append(node_path_m4_i)
	edge_path_m4.append(edge_path_m4_i)
	distance_path_m4.append(distance_path_m4_i)

i_m4 = find_best_iteration(distance_cum_m4)

distance_cum_m4_best = distance_cum_m4[i_m4]
node_path_m4_best = node_path_m4[i_m4]
edge_path_m4_best = edge_path_m4[i_m4]
distance_path_m4_best = distance_path_m4[i_m4]

route_eff_m4, extra_distance_m4, most_explored_edge_m4, most_explored_n_m4 = calc_metrics(
	optimal_distance = optimal_distance,
	actual_distance = distance_cum_m4_best,
	edge_path = edge_path_m4_best
	)

progress_m4 = []
for i in range(len(edge_path_m4_best)):
	progress_m4.append(len(set(edge_path_m4_best[:i])) / edges)

fig_m4, ax_m4 = plt.subplots(1, 1)
ax_m4.hist(distance_cum_m4)
ax_m4.set_xlabel("Distance Travelled")
ax_m4.set_ylabel("Count")
fig_m4.suptitle("Random w/o Backtracking Path Generation", fontsize = 16)
plt.title("Average Distance = " + str(round(sum(distance_cum_m4) / n, 1)) + " miles", loc = "left")
plt.savefig("./viz/m4_dist_hist.png")
plt.close()

progress_m4_df = pd.DataFrame({
	"distance": np.cumsum(np.array(distance_path_m4_best)),
	"pct_edges_complete": progress_m4})
fig_m4_progress, ax_m4_progress = plt.subplots(1, 1)
ax_m4_progress.plot(progress_m4_df["distance"], progress_m4_df["pct_edges_complete"])
ax_m4_progress.set_xlabel("Distance Travelled")
ax_m4_progress.set_ylabel("Proportion of Edges Traversed")
fig_m4_progress.suptitle("Random w/o Backtracking Path Generation", fontsize = 16)
plt.savefig("./viz/m4_progress.png")
plt.close()

print("--- Model 4: Random w/o Backtracking --------------------------------")
print("Distance:", round(distance_cum_m4_best, 2))
print("Route Efficiency:", round(route_eff_m4, 2))

# Model 5: Tree Search #######################################################

def postman_eagle(connections_dict, edges_n, edges_list, nodes_by_edge, graph, start = 0, shortest_path_flag = 1):

	current_loc = start
	distance_cum = 0
	node_path = [current_loc]
	edge_path = []
	distance_path = []

	w1 = 1
	w2 = 0.5
	w3 = 0.25

	while len(set(edge_path)) != edges_n:
		layer1_edges = [i[2] for i in connections_dict[current_loc]]
		layer1_nodes = [i[0] for i in connections_dict[current_loc]]
		layer2_edges = [[j[2] for j in connections_dict[i] if (j[0] != current_loc) & (j[2] not in layer1_edges)] for i in layer1_nodes]
		layer2_nodes = [[j[0] for j in connections_dict[i] if (j[0] != current_loc) & (j[0] not in layer1_nodes)] for i in layer1_nodes]
		layer3_intermediate_nodes = [[connections_dict[i] for i in j] for j in layer2_nodes]
		layer3_edges = []
		for i in layer3_intermediate_nodes:
			layer3_i = []
			for j in i:
				x = [k[2] for k in j if (k[0] != current_loc) & (k[0] not in layer1_nodes) & (k[0] not in layer2_nodes)]
				layer3_i.append(x)
			layer3_edges.append(layer3_i)

		s1 = [w1 if i not in edge_path else 0 for i in layer1_edges]
		s2 = [[w2 if j not in edge_path else 0 for j in i] for i in layer2_edges]
		s2 = [sum(i) for i in s2]
		s3 = []
		for i in layer3_edges:
			layer3_i_flat = [a for b in i for a in b]
			layer3_i_flat = [j for j in layer3_i_flat if j not in edge_path]
			n3_i = len(set(layer3_i_flat))
			s3.append(w3*n3_i)

		options_w = [s1[i] + s2[i] + s3[i] for i in range(len(s1))]

		max_option = pd.Series(options_w).max()
		options_w_max = pd.Series(options_w).loc[lambda x: x == max_option]
		most_options_i = pd.Series(options_w_max).sample().index[0]

		if max(options_w) != 0:
			path = connections_dict[current_loc][most_options_i]
			distance_cum += path[1]
			current_loc = path[0]
			node_path.append(current_loc)
			edge_path.append(path[2])
			distance_path.append(path[1])
		# this just sends to the next node not the entire path
		# thus, force_edge_destination chooses a new one the next iteration
		# put lines 605-609 in if statement?
		elif shortest_path_flag == 1:
			remaining_edges = [i for i in edges_list if i not in set(edge_path)]
			# maybe remove [0] and flatten/unnest?
			node_destinations = [nodes_by_edge[i] for i in remaining_edges]
			node_destinations = [i for j in node_destinations for i in j]
			# print(current_loc)
			# print(node_destinations)

			shortest_path = nx.multi_source_dijkstra(
				G = G,
				sources = node_destinations,
				target = current_loc,
				weight = "weight"
				)[1]
			shortest_path.reverse()
			del shortest_path[0]

			for i in shortest_path:
				for j in connections_dict[i]:
					if j[0] == current_loc:
						distance_cum += j[1]
						current_loc = i
						node_path.append(current_loc)
						edge_path.append(j[2])
						distance_path.append(j[1])

				
		else:
			path = random.choice(connections_dict[current_loc])
			distance_cum += path[1]
			current_loc = path[0]
			node_path.append(current_loc)
			edge_path.append(path[2])
			distance_path.append(path[1])

	return distance_cum, node_path, edge_path, distance_path

distance_cum_m5_full, node_path_m5_full, edge_path_m5_full, distance_path_m5_full = simulate(
	f = postman_eagle,
	n = n,
	connections_dict = connections,
	edges_n = edges,
	edges_list = edges_list,
	nodes_by_edge = nodes_by_edge,
	graph = G,
	start = 0,
	shortest_path_flag = 0
	)

distance_cum_m5 = []
node_path_m5 = []
edge_path_m5 = []
distance_path_m5 = []
for i in range(len(distance_cum_m5_full)):

	distance_cum_m5_i, node_path_m5_i, edge_path_m5_i, distance_path_m5_i = prune_path(
		node_path = node_path_m5_full[i],
		edge_path = edge_path_m5_full[i],
		distance_path = distance_path_m5_full[i]
		)
	distance_cum_m5.append(distance_cum_m5_i)
	node_path_m5.append(node_path_m5_i)
	edge_path_m5.append(edge_path_m5_i)
	distance_path_m5.append(distance_path_m5_i)

i_m5 = find_best_iteration(distance_cum_m5)

distance_cum_m5_best = distance_cum_m5[i_m5]
node_path_m5_best = node_path_m5[i_m5]
edge_path_m5_best = edge_path_m5[i_m5]
distance_path_m5_best = distance_path_m5[i_m5]

route_eff_m5, extra_distance_m5, most_explored_edge_m5, most_explored_n_m5 = calc_metrics(
	optimal_distance = optimal_distance,
	actual_distance = distance_cum_m5_best,
	edge_path = edge_path_m5_best
	)

progress_m5 = []
for i in range(len(edge_path_m5_best)):
	progress_m5.append(len(set(edge_path_m5_best[:i])) / edges)

fig_m5, ax_m5 = plt.subplots(1, 1)
ax_m5.hist(distance_cum_m5)
ax_m5.set_xlabel("Distance Travelled")
ax_m5.set_ylabel("Count")
fig_m5.suptitle("Tree Search Path Generation", fontsize = 16)
plt.title("Average Distance = " + str(round(sum(distance_cum_m5) / n, 1)) + " miles", loc = "left")
plt.savefig("./viz/m5_dist_hist.png")
plt.close()

progress_m5_df = pd.DataFrame({
	"distance": np.cumsum(np.array(distance_path_m5_best)),
	"pct_edges_complete": progress_m5})
fig_m5_progress, ax_m5_progress = plt.subplots(1, 1)
ax_m5_progress.plot(progress_m5_df["distance"], progress_m5_df["pct_edges_complete"])
ax_m5_progress.set_xlabel("Distance Travelled")
ax_m5_progress.set_ylabel("Proportion of Edges Traversed")
fig_m5_progress.suptitle("Tree Search Path Generation", fontsize = 16)
plt.savefig("./viz/m5_progress.png")
plt.close()

print("--- Model 5: Tree Search --------------------------------------------")
print("Distance:", round(distance_cum_m5_best, 2))
print("Route Efficiency:", round(route_eff_m5, 2))

print("Best route:")
print(node_path_m5_best)
