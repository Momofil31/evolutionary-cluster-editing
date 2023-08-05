import random
from utils import (
    compute_cost_of_isolating_node,
    compute_cost_of_adding_node_to_cluster,
    compute_cluster_mapping,
)
from evaluation import evaluate


def reduce_clusters(individual):
    # Reduce the number of cluster indexes used.
    # This is done to avoid increasing the search space, by dropping unused cluster indexes.
    # For example if the individual is [10, 2, 3, 54, 54 12, 3], the search space would be in [0, 54]
    # and the individual would become [0, 1, 2, 3, 3, 4, 1] with a search space in [0, 4]

    unique_clusters = list(set(individual))
    cluster_mapping = {cluster: idx for idx, cluster in enumerate(unique_clusters)}
    individual[:] = [cluster_mapping[cluster] for cluster in individual]
    individual.cluster_mapping = compute_cluster_mapping(individual)
    return individual


def node_isolation(individual, subset_ratio: float = 0.25):
    # Each node of the random subset is isolated from every other cluster
    # i.e. a new cluster is created for each node of the subset

    # draw random subset of subset_ratio * individual.num_nodes integers from [0, individual.num_nodes]
    num_nodes = len(individual)
    subset_size = int(subset_ratio * num_nodes)
    subset_idxs = random.sample(range(num_nodes), subset_size)

    cluster_indexes = set(individual)
    available_indexes = set(range(max(cluster_indexes))) - (cluster_indexes)

    if len(available_indexes) == num_nodes:
        # all nodes are already isolated
        return (individual,)

    for node_idx in subset_idxs:
        old_cluster = individual[node_idx]
        if len(available_indexes):
            new_cluster = random.choice(list(available_indexes))
            individual[node_idx] = new_cluster
            available_indexes.remove(individual[node_idx])
            cluster_indexes.add(individual[node_idx])
        else:
            new_cluster = max(cluster_indexes) + 1
            individual[node_idx] = new_cluster
            cluster_indexes.add(individual[node_idx])

        if individual.fitness.valid:
            # update fitness using cost of moving node to new cluster
            fitness = individual.fitness.values[0]

            fitness += compute_cost_of_isolating_node(
                node_idx,
                old_cluster,
                individual.graph,
                individual.cluster_mapping,
            )
            fitness += compute_cost_of_adding_node_to_cluster(
                node_idx,
                new_cluster,
                individual.graph,
                individual.cluster_mapping,
            )
            individual.fitness.values = (fitness,)
        if new_cluster != old_cluster:
            individual.cluster_mapping[new_cluster].add(node_idx)
            individual.cluster_mapping[old_cluster].remove(node_idx)
            if len(individual.cluster_mapping[old_cluster]) == 0:
                del individual.cluster_mapping[old_cluster]

    return (individual,)


def node_mover(individual, subset_ratio: float = 0.25):
    # Each node of the random subset is moved to a random existing cluster

    clusters = set(individual)
    num_clusters = len(clusters)

    # Draw random subset of subset_ratio*len(individual) integers from [0, len(individual)]
    num_nodes = len(individual)
    subset_size = int(subset_ratio * num_nodes)
    subset_idxs = random.sample(range(num_nodes), subset_size)

    for node_idx in subset_idxs:
        old_cluster = individual[node_idx]
        new_cluster = random.choice(list(set(individual)))

        if individual.fitness.valid:
            # update fitness using cost of moving node to new cluster
            fitness = individual.fitness.values[0]
            fitness += compute_cost_of_isolating_node(
                node_idx,
                old_cluster,
                individual.graph,
                individual.cluster_mapping,
            )
            fitness += compute_cost_of_adding_node_to_cluster(
                node_idx,
                new_cluster,
                individual.graph,
                individual.cluster_mapping,
            )
            individual.fitness.values = (fitness,)
        individual[node_idx] = new_cluster

        if new_cluster != old_cluster:
            individual.cluster_mapping[new_cluster].add(node_idx)
            individual.cluster_mapping[old_cluster].remove(node_idx)
            if len(individual.cluster_mapping[old_cluster]) == 0:
                del individual.cluster_mapping[old_cluster]

    # assert that the number of clusters did not increase
    assert (
        len(set(individual)) == len(individual.cluster_mapping.keys())
        and len(set(individual)) <= num_clusters
    )
    return (individual,)


def mutRemoveCliques(individual, subset_ratio: float = 0.30):
    """Randomly chose a subset of clusters and remove them from the individual"""
    clusters = set(individual)
    num_clusters = len(clusters)
    if num_clusters == 1:
        return (individual,)
    else:
        subset_size = int(subset_ratio * num_clusters)
        subset_idxs = random.sample(clusters, subset_size)
        if len(subset_idxs):
            for cluster_to_remove in subset_idxs:
                for node_idx, cluster in enumerate(individual):
                    if cluster == cluster_to_remove:
                        new_cluster = random.choice(
                            list(set(individual) - {cluster_to_remove})
                        )
                        if individual.fitness.valid:
                            # update fitness using cost of moving node to new cluster
                            fitness = individual.fitness.values[0]
                            fitness += compute_cost_of_isolating_node(
                                node_idx,
                                cluster_to_remove,
                                individual.graph,
                                individual.cluster_mapping,
                            )
                            fitness += compute_cost_of_adding_node_to_cluster(
                                node_idx,
                                new_cluster,
                                individual.graph,
                                individual.cluster_mapping,
                            )
                            individual.fitness.values = (fitness,)
                        individual[node_idx] = new_cluster
                        individual.cluster_mapping[new_cluster].add(node_idx)
                        individual.cluster_mapping[cluster].remove(node_idx)
                        if len(individual.cluster_mapping[cluster]) == 0:
                            del individual.cluster_mapping[cluster]

            assert (
                len(set(individual)) == len(individual.cluster_mapping.keys())
                and len(set(individual)) < num_clusters
            )
        return (individual,)


def mutate_moveOrIsolateOrRemoveCliques(
    individual,
    movepb: float = 0.3,
    isolatepb: float = 0.3,
    removepb: float = 0.4,
    subset_ratio: float = 0.25,
):
    # individual = reduce_clusters(individual)
    # assert movepb + isolatepb + removepb == 1
    op_choice = random.random()
    if op_choice < movepb:
        return node_mover(individual, subset_ratio)
    elif movepb <= op_choice < movepb + isolatepb:
        return node_isolation(individual, subset_ratio)
    elif movepb + isolatepb <= op_choice <= movepb + isolatepb + removepb:
        return mutRemoveCliques(individual, subset_ratio)
    else:  # no mutation
        return (individual,)
