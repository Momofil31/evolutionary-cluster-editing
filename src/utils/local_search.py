import random
import heapq

from utils.graph import (
    compute_cluster_mapping,
    compute_cost_of_adding_node_to_cluster,
    compute_cost_of_isolating_node,
)


def move_to_best_cluster(
    node,
    individual,
    graph,
    consider_original_cluster=False,
    consider_isolation=False,
):
    num_edges = []  # num_edges is a min heap of (cost, cluster) tuples
    cur_cluster = individual[node]
    isolation_cost = compute_cost_of_isolating_node(
        node, cur_cluster, graph, individual.cluster_mapping
    )

    for cluster in individual.cluster_mapping.keys():
        if consider_original_cluster and cluster == cur_cluster:
            continue

        cost = individual.fitness.values[0]
        cost += isolation_cost
        cost += compute_cost_of_adding_node_to_cluster(
            node, cluster, graph, individual.cluster_mapping
        )

        heapq.heappush(num_edges, (cost, cluster))

    if consider_isolation:
        cost = individual.fitness.values[0]
        cost += isolation_cost
        cluster_indexes = set(individual)
        available_indexes = set(range(max(cluster_indexes))) - cluster_indexes

        if len(available_indexes):
            cluster = random.choice(list(available_indexes))
            heapq.heappush(num_edges, (cost, cluster))
            available_indexes.remove(cluster)
        else:
            cluster = max(cluster_indexes) + 1
            heapq.heappush(num_edges, (cost, cluster))
            cluster_indexes.add(cluster)

    if num_edges:
        # assign node to the cluster with the lowest number of added/removed edges
        best_cost, best_cluster = heapq.heappop(num_edges)
        individual[node] = best_cluster
        if best_cluster != cur_cluster:
            individual.cluster_mapping[best_cluster].add(node)
            individual.cluster_mapping[cur_cluster].remove(node)
            if len(individual.cluster_mapping[cur_cluster]) == 0:
                del individual.cluster_mapping[cur_cluster]
        else:
            assert best_cost == individual.fitness.values[0]
        individual.fitness.values = (best_cost,)
    return individual


def label_propagation(individual, graph, subset=1.0):
    # Label Propagation Algorithm
    # iterate nodes in random order
    # assign each node to the cluster that minimizes the number of modified edges.
    # a possible assignment is also to a new cluster

    subset_size = int(subset * len(individual))
    nodes = list(set(random.sample(range(len(individual)), subset_size)))

    for node in nodes:
        individual = move_to_best_cluster(
            node,
            individual,
            graph,
            consider_original_cluster=True,
            consider_isolation=True,
        )

    return (individual,)


def clique_removal(individual, graph, subset=1.0, max_remove_clique_size=5):
    clusters = list(individual.cluster_mapping.keys())
    subset_size = int(subset * len(clusters))
    clusters = random.sample(clusters, subset_size)

    for cluster_to_remove in clusters:
        if len(individual.cluster_mapping[cluster_to_remove]) <= max_remove_clique_size:
            cluster_nodes = list(individual.cluster_mapping[cluster_to_remove])
            for node in cluster_nodes:
                if node in individual.cluster_mapping[cluster_to_remove]:
                    individual = move_to_best_cluster(
                        node,
                        individual,
                        graph,
                        consider_original_cluster=False,
                        consider_isolation=True,
                    )

    return (individual,)
