from pathlib import Path
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import src.utils.utils as utils
from src.utils.graph import compute_cluster_mapping


class GraphModule:
    def __init__(self, data_dir: str, bounds_dir: str) -> None:
        self.data_dir = data_dir
        self.bounds_dir = bounds_dir
        self.graph = None
        self.components = None
        self.l_bound = None
        self.u_bound = None

    def load_graph(self, input: str, bounds: str):
        """Load graph from file
        input is a file
        N M
        u1 v1
        u2 v2
        ...
        uM vM
        """

        self.graph = {}
        with open(Path(self.data_dir, input), "r") as f:
            N, M = [int(x) for x in f.readline().split()]
            for i in range(N):
                self.graph[i] = set()
            for i in range(M):
                u, v = [int(x) for x in f.readline().split()]
                self.graph[u].add(v)
                self.graph[v].add(u)
        self.components = self.compute_clusters()

        with open(Path(self.bounds_dir, bounds), "r") as f:
            self.l_bound, self.u_bound = [int(x) for x in f.readline().split()]

    def compute_clusters(self):
        """Find connected components in the graph"""
        clustering = [-1] * len(self.graph)
        cluster_id = 0
        for i in range(len(self.graph)):
            if clustering[i] == -1:
                clustering[i] = cluster_id
                cluster_id += 1
                queue = [i]
                while len(queue) > 0:
                    u = queue.pop(0)
                    for v in self.graph[u]:
                        if clustering[v] == -1:
                            clustering[v] = clustering[u]
                            queue.append(v)
        return clustering

    def compute_edits(self, individual):
        """Compute added and removed edges from the graph given an individual."""
        added_edges = set()
        removed_edges = set()

        for i in range(len(self.graph)):
            for j in range(i + 1, len(self.graph)):
                if individual[i] == individual[j]:
                    if j not in self.graph[i] and i not in self.graph[j]:
                        added_edges.add((i, j))
                else:
                    if j in self.graph[i] and i in self.graph[j]:
                        removed_edges.add((i, j))

        return added_edges, removed_edges

    def is_valid_solution(self):
        """Check if the solution is valid
        For each connected component check that it is a clique.
        """
        cluster_assignment = self.compute_clusters()
        components = compute_cluster_mapping(cluster_assignment)
        for _, component in components.items():
            for node in component:
                if len(self.graph[node]) != len(component) - 1:
                    return False
        return True

    def apply_edits(self, cluster_assignments):
        """Apply edits to the graph given a cluster assignment"""
        added, removed = self.compute_edits(cluster_assignments)

        for edge in added:
            self.graph[edge[0]].add(edge[1])
            self.graph[edge[1]].add(edge[0])

        for edge in removed:
            self.graph[edge[0]].remove(edge[1])
            self.graph[edge[1]].remove(edge[0])

        self.components = self.compute_clusters()
