from utils import compute_cluster_mapping


# Define the Evaluation Function
# Don't need to check feasibility because with this formulation, all individuals are always feasible by construction
# I assume that a cluster is a clique since i compute the fitness based on the number of edges that need to be added or removed
# to build such clusters.
def evaluate(individual):
    # Objective Function is the number of modifications (+ or -) to the input graph
    # Complexity is O(n^2) where n is the number of nodes

    individual.cluster_mapping = compute_cluster_mapping(individual)
    graph = individual.graph

    added_edges = 0
    removed_edges = 0

    for i in range(len(graph)):
        for j in range(i + 1, len(graph)):
            if individual[i] == individual[j]:
                if j not in graph[i] and i not in graph[j]:
                    added_edges += 1
            else:
                if j in graph[i] and i in graph[j]:
                    removed_edges += 1

    objective_value = added_edges + removed_edges

    return (float(objective_value),)
