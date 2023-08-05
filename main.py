import random
import numpy as np
from collections import defaultdict
from deap import base, creator, tools, algorithms
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from mutation import mutate_moveOrIsolateOrRemoveCliques, reduce_clusters
from graph import (
    load_graph,
    compute_clusters,
    compute_solution,
    clique_removal,
    label_propagation,
)
from evaluation import evaluate
from utils import compute_cluster_mapping
from crossover import cxCommonCluster

# Problem Constants
MAX_CLUSTERS = 0
NUM_NODES = 0
NGEN = 50  # Number of generations
POP_SIZE = 200  # Population size
LAMBDA = 50  # Number of offsprings to generate
MU = POP_SIZE  # Number of individuals to select for the next generation

# Subset of the individual nodes to apply label propagation to
LABEL_PROPAGATION_SUBSET = 1.

# Subset of the individual cluster to apply clique removal to
CLIQUE_REMOVAL_SUBSET = 1.

# Probabilities
CXPB = 0.5  # Crossover probability
MUTPB = 0.5  # Mutation probability
LOCAL_SEARCH_PB = 1.0

# Flags
ENABLE_LOCAL_SEARCH = False
DRAW_GRAPH = True
USE_SEED_INDIVIDUAL = False

# Input
input = "input16.txt"
data_folder = "input"

def varAnd(population, toolbox, lambda_, cxpb, mutpb):
    offspring = []

    for _ in range(lambda_):
        if random.random() < cxpb:  # Apply crossover
            ind1, ind2 = [toolbox.clone(i) for i in random.sample(population, 2)]
            out = toolbox.mate(ind1, ind2)
            if type(out) is tuple:
                off1, off2 = out
            else:
                off1 = out
            del off1.fitness.values
            off1.cluster_mapping = compute_cluster_mapping(off1)

            if random.random() < mutpb:  # Apply mutation
                # mutation already updates the cluster mapping
                (off1,) = toolbox.mutate(off1)
                del ind.fitness.values
            offspring.append(off1)
        else:  # Apply reproduction
            offspring.append(random.choice(population))

    return offspring


def varOr(population, toolbox, lambda_, cxpb, mutpb):
    offspring = [toolbox.clone(ind) for ind in population]

    assert (cxpb + mutpb) <= 1.0, (
        "The sum of the crossover and mutation probabilities must be smaller "
        "or equal to 1.0."
    )

    offspring = []
    for _ in range(lambda_):
        op_choice = random.random()
        if op_choice < cxpb:  # Apply crossover
            ind1, ind2 = [toolbox.clone(i) for i in random.sample(population, 2)]
            out = toolbox.mate(ind1, ind2)
            if type(out) is tuple:
                off1, off2 = out
            else:
                off1 = out
            del off1.fitness.values
            off1.cluster_mapping = compute_cluster_mapping(off1)
            offspring.append(off1)
        elif op_choice < cxpb + mutpb:  # Apply mutation
            ind = toolbox.clone(random.choice(population))
            # mutation already updates the cluster mapping
            (ind,) = toolbox.mutate(ind)
            del ind.fitness.values
            offspring.append(ind)
        else:  # Apply reproduction
            offspring.append(random.choice(population))

    return offspring


def mu_plus_lambda(graph, mu, lambda_, verbose=True, seed_individual=None):
    # Set up Statistics and Hall of Fame
    hof = tools.ParetoFront()
    fit_stats = tools.Statistics(lambda ind: ind.fitness.values)
    num_clusters_stats = tools.Statistics(lambda ind: len(ind.cluster_mapping.keys()))
    stats = tools.MultiStatistics(fitness=fit_stats, num_clusters=num_clusters_stats)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    # Initialize the Population with half og the individuals being the seed_individual
    population = toolbox.population(n=POP_SIZE)
    if seed_individual is not None:
        for idx in range(len(population) // 2):
            population[idx] = toolbox.clone(seed_individual)

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    for ind in population:
        ind.cluster_mapping = compute_cluster_mapping(ind)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if hof is not None:
        hof.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Run the Evolutionary Algorithm
    try:
        for gen in tqdm(range(NGEN)):
            population = toolbox.select(population, k=len(population))
            # offspring = varOr(population, toolbox, lambda_, CXPB, MUTPB)
            offspring = varAnd(population, toolbox, lambda_, CXPB, MUTPB)

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fits = toolbox.map(toolbox.evaluate, invalid_ind)

            for fit, ind in zip(fits, invalid_ind):
                ind.fitness.values = fit

            # local search (very computationallty expensive)
            if ENABLE_LOCAL_SEARCH:
                for ind in tqdm(offspring, "Local search", leave=False):
                    if random.random() < LOCAL_SEARCH_PB:
                        label_propagation(ind, graph, LABEL_PROPAGATION_SUBSET)
                        clique_removal(ind, graph, CLIQUE_REMOVAL_SUBSET)

            if hof is not None:
                hof.update(offspring)

            # Select the next generation population
            population[:] = toolbox.select(population + offspring, mu)

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)
    except KeyboardInterrupt:
        print("Loop Interrupted")
    return hof


if __name__ == "__main__":
    # Load Data
    graph = load_graph(f"{data_folder}/{input}")
    # need to estimate a good value for MAX_CLUSTERS
    NUM_NODES = len(graph)
    print("NUM_NODES: ", NUM_NODES)

    components = compute_clusters(graph)

    # Create the Fitness and Individual classes
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimization problem
    creator.create(
        "Individual",
        list,
        fitness=creator.FitnessMin,
        max_clusters=MAX_CLUSTERS,
        num_nodes=NUM_NODES,
        cluster_mapping=defaultdict(set),
        graph=graph,
    )

    ind = creator.Individual(components)
    ind.cluster_mapping = compute_cluster_mapping(ind)

    print(evaluate(ind))
    added, removed = compute_solution(ind, ind.graph)
    print("Added edges: ", len(added))
    print("Removed edges: ", len(removed))

    # first compute the initial number of clusters for the unfeasible input graph
    num_clusters = max(ind) + 1
    print("Num_clusters: ", num_clusters)

    # Add some variance to this value
    MAX_CLUSTERS = int(NUM_NODES * 0.1)
    # MAX_CLUSTERS = 50
    print("MAX_CLUSTERS: ", MAX_CLUSTERS)

    # Initialize the Toolbox
    toolbox = base.Toolbox()

    # Define the Decision Variables
    toolbox.register(
        "attr_int", random.randint, 0, MAX_CLUSTERS - 1
    )  # Integer values [0, max_clusters]
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_int,
        n=NUM_NODES,
    )

    # Define the Population
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register the Evaluation Function and Constraints
    toolbox.register("evaluate", evaluate)

    # Define the Genetic Operators
    # toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mate", cxCommonCluster, toolbox=toolbox, indpb=0.5)
    toolbox.register(
        "mutate",
        mutate_moveOrIsolateOrRemoveCliques,
        movepb=0.3,
        isolatepb=0.1,
        removepb=0.6,
    )
    toolbox.register("select", tools.selNSGA2)

    ind.fitness.values = toolbox.evaluate(ind)
    seed_ind = None
    if USE_SEED_INDIVIDUAL:
        seed_ind = ind
    hof = mu_plus_lambda(graph, MU, LAMBDA, verbose=True, seed_individual=seed_ind)

    if len(hof) > 0:
        best_individual = hof[0]
        print("Best individual: ", best_individual, best_individual.fitness)
        print("Num clusters: ", len(set(best_individual)))
        if DRAW_GRAPH:
            nx_final_graph = nx.Graph(graph)
            added, removed = compute_solution(best_individual, graph)

            nx_final_graph.add_edges_from(added)
            nx_final_graph.remove_edges_from(removed)
            nx.draw_networkx(nx_final_graph)
            plt.show()
    else:
        print("No feasible solution found")
