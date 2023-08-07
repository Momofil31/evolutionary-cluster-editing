import random
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import pylogger
from src.utils.graph import compute_cluster_mapping

log = pylogger.get_pylogger(__name__)


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
                # no need to delete fitness values since crossover already invalidates them
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
