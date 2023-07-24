import random
from deap import base, creator, tools
from evaluation import evaluate
from mutation import mutate_moveOrIsolate
import networkx as nx
import matplotlib.pyplot as plt

MAX_REMOVE_CLIQUE_SIZE = 3


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


def label_propagation(individual, graph):
    # Label Propagation Algorithm
    # iterate nodes in random order
    # assign each node to the cluster that minimizes the number of modified edges.
    # a possible assignment is also to a new cluster

    nodes = list(range(len(individual)))
    random.shuffle(nodes)
    clusters = set(individual)
    cluster_mapping = {cluster: [] for cluster in clusters}
    for node, cluster in enumerate(individual):
        cluster_mapping[cluster].append(node)

    for node in nodes:
        # compute the number of edges that would be added/removed if node was assigned to each cluster
        # if node was assigned to a new cluster, the number of added/removed edges is the same
        # the cluster with the lowest index is chosen in case of a tie
        num_edges = []
        for cluster in clusters:
            cost = individual.fitness.values[0]
            # cost of isolating the node
            # number of neighbors in the old cluster - number of non-neighbors in the old cluster
            x = 0
            y = 0
            old_cluster = individual[node]
            for neighbor in graph[node]:
                if individual[neighbor] == old_cluster:
                    x += 1
            for cluster_node in cluster_mapping[old_cluster]:
                if cluster_node not in graph[node] and cluster_node != node:
                    y += 1
            cost += x - y
            # cost of adding the node to the new cluster
            # number of non-neighbors in the new cluster - number of neighbors in the old cluster
            x = 0
            y = 0
            for cluster_node in cluster_mapping[cluster]:
                if cluster_node not in graph[node] and cluster_node != node:
                    x += 1

            for cluster_node in cluster_mapping[old_cluster]:
                if cluster_node in graph[node] and cluster_node != node:
                    y += 1
            cost += x - y

            num_edges.append((cost, cluster))

        # try to assign node to a new cluster
        # assigning node to a new cluster means that i only need to remove edges
        # cost of isolating the node
        # number of neighbors in the old cluster - number of non-neighbors in the old cluster
        x = 0
        y = 0
        cost = individual.fitness.values[0]
        old_cluster = individual[node]
        for neighbor in graph[node]:
            if individual[neighbor] == old_cluster:
                x += 1
        for cluster_node in cluster_mapping[old_cluster]:
            if cluster_node not in graph[node] and cluster_node != node:
                y += 1
        cost += x - y

        num_edges.append((cost, max(clusters) + 1))

        # sort the clusters by the number of added/removed edges
        num_edges.sort()
        # assign node to the cluster with the lowest number of added/removed edges
        individual[node] = num_edges[0][1]
        individual.fitness.values = evaluate(individual, graph)

    return (individual,)


def compute_cost_of_isolating_node(node, individual, graph, cluster_mapping):
    # cost of isolating the node
    # number of neighbors in the old cluster - number of non-neighbors in the old cluster
    x = 0
    y = 0
    old_cluster = individual[node]
    for neighbor in graph[node]:
        if individual[neighbor] == old_cluster:
            x += 1
    for cluster_node in cluster_mapping[old_cluster]:
        if cluster_node not in graph[node] and cluster_node != node:
            y += 1
    cost = x - y
    return cost


def compute_cost_of_adding_node_to_cluster(
    node, individual, new_cluster, graph, cluster_mapping
):
    # cost of adding the node to the new cluster
    # number of non-neighbors in the new cluster - number of neighbors in the old cluster
    x = 0
    y = 0
    old_cluster = individual[node]
    for cluster_node in cluster_mapping[new_cluster]:
        if cluster_node not in graph[node] and cluster_node != node:
            x += 1

    for cluster_node in cluster_mapping[old_cluster]:
        if cluster_node in graph[node] and cluster_node != node:
            y += 1
    cost = x - y
    return cost


def move_to_best_cluster(
    node, individual, graph, cluster_mapping, consider_original_cluster=False
):
    num_edges = []
    cur_cluster = individual[node]
    for cluster in clusters:
        if consider_original_cluster and cluster == cur_cluster:
            continue

        cost = individual.fitness.values[0]
        cost += compute_cost_of_isolating_node(node, individual, graph, cluster_mapping)
        cost += compute_cost_of_adding_node_to_cluster(
            node, individual, cluster, graph, cluster_mapping
        )

        num_edges.append((cost, cluster))

    if num_edges:
        # sort the clusters by the number of added/removed edges
        num_edges.sort()
        # assign node to the cluster with the lowest number of added/removed edges
        individual[node] = num_edges[0][1]
    return individual


def clique_removal(individual, graph):
    tool = base.Toolbox()
    clusters = list(set(individual))
    random.shuffle(clusters)
    cluster_mapping = {cluster: [] for cluster in clusters}
    for node, cluster in enumerate(individual):
        cluster_mapping[cluster].append(node)

    for cluster_to_remove in clusters:
        individual_new = tool.clone(individual)
        if len(cluster_mapping[cluster_to_remove]) <= MAX_REMOVE_CLIQUE_SIZE:
            for node in cluster_mapping[cluster_to_remove]:
                individual_new = move_to_best_cluster(
                    node,
                    individual_new,
                    graph,
                    cluster_mapping,
                    consider_original_cluster=False,
                )

            individual_new.fitness.values = evaluate(individual_new, graph)
            if individual_new.fitness.values[0] > individual.fitness.values[0]:
                individual = individual_new

                cluster_mapping = {cluster: [] for cluster in clusters}
                for node, cluster in enumerate(individual):
                    cluster_mapping[cluster].append(node)
    return (individual,)


if __name__ == "__main__":
    random.seed(64)
    # Testing
    graph = load_graph("input/input1.txt")
    clusters = compute_clusters(graph)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimization problem
    creator.create(
        "Individual",
        list,
        fitness=creator.FitnessMin,
        max_clusters=max(clusters) + 1,
        num_nodes=len(graph),
    )

    # ind = creator.Individual([i for i in range(len(graph))])
    ind = creator.Individual(clusters)

    toolbox = base.Toolbox()
    # Register the Evaluation Function and Constraints
    toolbox.register("evaluate", evaluate, graph=graph)
    toolbox.register("mutate", mutate_moveOrIsolate, indpb=0.5, subset_ratio=0.25)

    ind.fitness.values = toolbox.evaluate(ind)
    print("Initial individual", ind, ind.fitness.values[0])

    best_fitness = ind.fitness.values[0]
    best_ind = ind
    for i in range(1000):
        # perturb
        toolbox.mutate(ind)
        ind.fitness.values = toolbox.evaluate(ind)
        # print("1.", ind, ind.fitness.values[0])

        # local search
        (ind,) = label_propagation(ind, graph)
        # print("2.", ind, ind.fitness.values[0])
        (ind,) = clique_removal(ind, graph)
        # print("3.", ind, ind.fitness.values[0])
        if best_ind.fitness.values[0] > ind.fitness.values[0]:
            best_ind = toolbox.clone(ind)
            print("New solution", ind, ind.fitness.values[0])
            print("***")
    print(best_ind, best_ind.fitness.values[0])

    nx_final_graph = nx.Graph(graph)
    added, removed = compute_solution(best_ind, graph)

    nx_final_graph.add_edges_from(added)
    nx_final_graph.remove_edges_from(removed)
    nx.draw_networkx(nx_final_graph)
    plt.show()
