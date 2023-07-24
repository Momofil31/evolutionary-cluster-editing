import random
from collections import defaultdict
from deap import base, creator, tools, algorithms
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from mutation import mutate_moveOrIsolate, reduce_clusters
from graph import (
    load_graph,
    compute_clusters,
    compute_solution,
    clique_removal,
    label_propagation,
)
from evaluation import evaluate
from utils import compute_cluster_mapping

# Problem Constants
# Modify these according to your specific problem
NUM_VARIABLES = 5
MAX_CLUSTERS = 0
NUM_NODES = 0
NGEN = 10  # Number of generations
POP_SIZE = 10
CXPB = 0.5  # Crossover probability
MUTPB = 0.2  # Mutation probability
ENABLE_LOCAL_SEARCH = False
DRAW_GRAPH = False

input = "input18.txt"
data_folder = "input"


# Define the Algorithm
def main(seed_individual, graph):
    # Initialize the Population with half og the individuals being the seed_individual
    population = toolbox.population(n=POP_SIZE)
    for idx in range(len(population) // 2):
        population[idx] = toolbox.clone(seed_individual)

    # Set up Statistics and Hall of Fame
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", min)
    hof = tools.ParetoFront()

    # Run the Evolutionary Algorithm

    for gen in tqdm(range(NGEN)):
        offspring = [toolbox.clone(ind) for ind in population]

        # Apply crossover and mutation on the offspring
        for i in range(1, len(offspring), 2):
            if random.random() < CXPB:
                offspring[i - 1], offspring[i] = toolbox.mate(
                    offspring[i - 1], offspring[i]
                )
                del offspring[i - 1].fitness.values, offspring[i].fitness.values

        # update cluster mappings after crossover
        # TODO: develop an efficient and "local searchy" crossover operator
        for off in offspring:
            off.cluster_mapping = compute_cluster_mapping(off)

        for i in range(len(offspring)):
            if random.random() < MUTPB:
                (offspring[i],) = toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fits = toolbox.map(toolbox.evaluate, invalid_ind)

        for fit, ind in zip(fits, invalid_ind):
            ind.fitness.values = fit

        # local search (very computationallty expensive)
        if ENABLE_LOCAL_SEARCH:
            for ind in tqdm(offspring):
                if random.random() < 0.1:
                    label_propagation(ind, graph)
                    clique_removal(ind, graph)
        population = toolbox.select(offspring, k=len(population))

        hof.update(population)
        record = stats.compile(population)
        # print(f"Generation {gen+1}: {record}")

    # # Print the Best Individuals
    # print("\nBest Individuals:")
    # for ind in hof:
    #     print(ind, ind.fitness.values)

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

    print(evaluate(ind))
    added, removed = compute_solution(ind, ind.graph)
    print("Added edges: ", len(added))
    print("Removed edges: ", len(removed))

    # first compute the initial number of clusters for the unfeasible input graph
    num_clusters = max(ind) + 1
    print("Num_clusters: ", num_clusters)

    # Add some variance to this value
    MAX_CLUSTERS = NUM_NODES
    print("MAX_CLUSTERS: ", MAX_CLUSTERS)

    # Initialize the Toolbox
    toolbox = base.Toolbox()

    # Define the Decision Variables
    toolbox.register(
        "attr_int", random.randint, 0, MAX_CLUSTERS
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
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutate_moveOrIsolate, indpb=0.5, subset_ratio=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    ind.fitness.values = toolbox.evaluate(ind)

    hof = main(ind, graph)

    if len(hof) > 0:
        best_individual = hof[0]
        print("Best individual: ", best_individual, best_individual.fitness)
        if DRAW_GRAPH:
            nx_final_graph = nx.Graph(graph)
            added, removed = compute_solution(best_individual, graph)

            nx_final_graph.add_edges_from(added)
            nx_final_graph.remove_edges_from(removed)
            nx.draw_networkx(nx_final_graph)
            plt.show()
    else:
        print("No feasible solution found")