import random
from deap import tools
from utils import find_common_clusters


def cxCommonCluster(ind1, ind2, toolbox, indpb=0.5):
    common_clusters = find_common_clusters(ind1, ind2)
    offspring = toolbox.clone(ind1)
    offspring[:] = [None] * len(ind1)

    # Copy the common clusters from one parent to the offspring
    for cluster_id, nodes in common_clusters.items():
        for node in nodes:
            # or ind2[node], it doesn't matter since nodes are the same in both parents
            offspring[node] = ind1[node]

    # For the remaining nodes, randomly chose from ind1 or ind2
    for i in range(len(ind1)):
        if offspring[i] is None:
            if random.random() < indpb:
                offspring[i] = ind1[i]
            else:
                offspring[i] = ind2[i]

    return offspring
