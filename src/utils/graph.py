from collections import defaultdict
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


def compute_cost_of_isolating_node(node, old_cluster, graph, cluster_mapping):
    # cost of isolating the node
    # number of neighbors in the old cluster - number of non-neighbors in the old cluster
    old_cluster_nodes = set(cluster_mapping[old_cluster])
    neighbors_in_old_cluster = len(set(old_cluster_nodes).intersection(graph[node]))
    non_neighbors_in_old_cluster = len(
        old_cluster_nodes.difference(graph[node] | {node})
    )
    cost = neighbors_in_old_cluster - non_neighbors_in_old_cluster
    return cost


def compute_cost_of_adding_node_to_cluster(node, new_cluster, graph, cluster_mapping):
    # cost of adding the node to the new cluster
    # number of non-neighbors in the new cluster - number of neighbors in the new cluster
    cluster_nodes = cluster_mapping[new_cluster]

    non_neighbors_in_new_cluster = len(
        set(cluster_nodes).difference(graph[node] | {node})
    )
    neighbors_in_new_cluster = len(set(cluster_nodes).intersection(graph[node]))
    cost = non_neighbors_in_new_cluster - neighbors_in_new_cluster
    return cost


def compute_cluster_mapping(individual):
    cluster_mapping = defaultdict(set)
    for node, cluster in enumerate(individual):
        cluster_mapping[cluster].add(node)
    return cluster_mapping


def find_common_clusters(ind1, ind2):
    """Returns a dictionary where keys = clusters and values = set of nodes in that cluster
    so that the nodes in the cluster are the same in both individuals"""

    common_cluster_ids = set(ind1) & set(ind2)

    common_clusters = defaultdict(set)
    for cluster_id in common_cluster_ids:
        nodes_a = ind1.cluster_mapping[cluster_id]
        nodes_b = ind2.cluster_mapping[cluster_id]
        if nodes_a == nodes_b:
            common_clusters[cluster_id] = nodes_a

    return common_clusters
