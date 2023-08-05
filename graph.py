import random
import heapq
from deap import base, creator, tools
from evaluation import evaluate
from tqdm.auto import tqdm

from mutation import mutate_moveOrIsolateOrRemoveCliques
import networkx as nx
import matplotlib.pyplot as plt
import copy
from collections import defaultdict
from utils import (
    compute_cost_of_adding_node_to_cluster,
    compute_cost_of_isolating_node,
    compute_cluster_mapping,
)

MAX_REMOVE_CLIQUE_SIZE = 5
ITERS = 50
MUTPB = 0.5

def load_graph(input):
    # input is a file
    # N M
    # u1 v1
    # u2 v2
    # ...
    # uM vM

    graph = {}
    with open(input, "r") as f:
        N, M = [int(x) for x in f.readline().split()]
        for i in range(N):
            graph[i] = set()
        for i in range(M):
            u, v = [int(x) for x in f.readline().split()]
            graph[u].add(v)
            graph[v].add(u)

    return graph


def compute_clusters(graph):
    # connected components algorithm
    clustering = [-1] * len(graph)
    cluster_id = 0
    for i in range(len(graph)):
        if clustering[i] == -1:
            clustering[i] = cluster_id
            cluster_id += 1
            queue = [i]
            while len(queue) > 0:
                u = queue.pop(0)
                for v in graph[u]:
                    if clustering[v] == -1:
                        clustering[v] = clustering[u]
                        queue.append(v)
    return clustering


def compute_solution(individual, graph):
    added_edges = set()
    removed_edges = set()

    for i in range(len(graph)):
        for j in range(i + 1, len(graph)):
            if individual[i] == individual[j]:
                if j not in graph[i] and i not in graph[j]:
                    added_edges.add((i, j))
            else:
                if j in graph[i] and i in graph[j]:
                    removed_edges.add((i, j))

    return added_edges, removed_edges


def check_solution(graph):
    """Check if the solution is valid
    For each connected component check that it is a clique.
    """
    cluster_assignment = compute_clusters(graph)
    components = compute_cluster_mapping(cluster_assignment)
    for component_idx, component in components.items():
        for node in component:
            if len(graph[node]) != len(component) - 1:
                return False
    return True


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


def clique_removal(individual, graph, subset=1.0):
    clusters = list(individual.cluster_mapping.keys())
    subset_size = int(subset * len(clusters))
    clusters = random.sample(clusters, subset_size)

    for cluster_to_remove in clusters:
        if len(individual.cluster_mapping[cluster_to_remove]) <= MAX_REMOVE_CLIQUE_SIZE:
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


if __name__ == "__main__":
    random.seed(64)
    # Testing
    graph = load_graph("input/input18.txt")
    components = compute_clusters(graph)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimization problem
    creator.create(
        "Individual",
        list,
        fitness=creator.FitnessMin,
        max_clusters=max(components) + 1,
        num_nodes=len(graph),
        cluster_mapping=defaultdict(set),
        graph=graph,
    )

    ind = creator.Individual([i for i in range(len(graph))])
    # ind = creator.Individual(components)

    toolbox = base.Toolbox()
    # Register the Evaluation Function and Constraints
    toolbox.register("evaluate", evaluate)
    toolbox.register("mutate", mutate_moveOrIsolateOrRemoveCliques)

    ind.cluster_mapping = compute_cluster_mapping(ind)
    ind.fitness.values = toolbox.evaluate(ind)
    print("Initial individual", ind, ind.fitness.values[0])

    best_fitness = ind.fitness.values[0]
    best_ind = toolbox.clone(ind)
    for i in tqdm(range(ITERS)):
        # perturb
        if random.random() < MUTPB:
            toolbox.mutate(ind)
        if not ind.fitness.valid:
            print("Invalid fitness", ind.fitness.values[0])
            ind.fitness.values = toolbox.evaluate(ind)
        # print("1.", ind, ind.fitness.values[0])

        # local search
        (ind,) = label_propagation(ind, graph)
        # print("2.", ind, ind.fitness.values[0])
        (ind,) = clique_removal(ind, graph)
        # print("3.", ind, ind.fitness.values[0])

        assert ind.fitness.valid
        if best_ind.fitness.values[0] > ind.fitness.values[0]:
            best_ind = toolbox.clone(ind)
            print(f"New solution at iter {i}:", best_ind.fitness.values[0])
            print("***")
    print(best_ind, best_ind.fitness.values[0])
    print("Num clusters: ", len(set(best_ind)))
    nx_final_graph = nx.Graph(graph)
    added, removed = compute_solution(best_ind, graph)

    print("Fitness:", evaluate(best_ind))
    # add and remove edges from graph
    for edge in added:
        graph[edge[0]].add(edge[1])
        graph[edge[1]].add(edge[0])

    for edge in removed:
        graph[edge[0]].remove(edge[1])
        graph[edge[1]].remove(edge[0])

    print("Solution valid:", check_solution(graph))
    print("Num clusters:", len(set(best_ind)))

    # nx_final_graph.add_edges_from(added)
    # nx_final_graph.remove_edges_from(removed)
    # nx.draw_networkx(nx_final_graph)
    # plt.show()
