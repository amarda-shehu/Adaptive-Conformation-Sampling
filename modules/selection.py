# selection.py
# author: Ahmed Bin Zaman
# since: 06/2018

"""Module for selecting next generation from current generation.

This module provides functions to select next generation from
current generation.

Available Functions:
- truncation: Selects next generation via elitism truncation selection
    for single objective.
- fitness_proportional: Selects next generation via fitness proportional
    selection for single objective.
- quaternary_tournament: Selects next generation via quaternary
    tournament selection for single objective.
- uniform_stochastic: Selects next generation via uniform stochastic
    selection for single objective.
"""

import math
import pyrosetta.toolbox.extract_coords_pose as pte
import numpy as np


def truncation(parent_population, child_population, parents_scores,
               children_scores, elitism_rate):
    """Selects next generation using elitism truncation selection for
    single objective.

    This function implements truncation selection while ensuring elitism
    to select a specific number of members for the next generation.

    Args:
        parent_population: A list containing members of parent
            population.
        child_population: A list containing members of offspring
            population.
        parents_scores: A list containing scores of each member of the
            parent population. The format is:
                [member 1 score, member 2 score, ....]
            The order of members has to be consistent with
            parent_population argument.
        children_scores: A list containing scores of each member of the
            offspring population. The format is:
                [member 1 score, member 2 score, ....]
            The order of members has to be consistent with
            child_population argument.
        elitism_rate: A float between 0 and 1 indicating the elitism
            percentage.

    Returns:
        A list of members for the next generation of population.
    """
    population_size = len(parent_population)
    population_indices = list(range(population_size))

    sorted_parents_indices = [x for _, x in sorted(zip(
        parents_scores, population_indices
    ))]
    sorted_parents_scores = sorted(parents_scores)

    # Slice parent population using elitism rate
    slice_index = int(population_size * elitism_rate)
    selected_parents_indices = sorted_parents_indices[:slice_index]
    selected_parents = [parent_population[i] for i in selected_parents_indices]

    combined_population = selected_parents + child_population
    combined_scores = sorted_parents_scores[:slice_index] + children_scores
    combined_population_indices = list(range(len(combined_population)))

    sorted_population_indices = [x for _, x in sorted(zip(
        combined_scores, combined_population_indices
    ))]

    selected_population_indices = sorted_population_indices[:population_size]

    # Truncate and return
    return [combined_population[i] for i in selected_population_indices]


def fitness_proportional(population, scores, next_gen_number, random_seed=42):
    """Selects next generation using fitness proportional selection for
    single objective.

    This function implements fitness proportional selection to select a
    specific number of members for the next generation.

    Args:
        population: A list containing members of current population with
            parents and children combined.
        scores: A list containing scores of each member of the
            population. The format is:
                [member 1 score, member 2 score, ....]
            The order of members has to be consistent with population
            argument.
        next_gen_number: An int indicating the number of members to be
            selected for the next generation.
        random_seed: An int indicating the seed of the random number
            generator.

    Returns:
        A list of members for the next generation of population.
    """

    np.random.seed(random_seed)

    score_array = np.array(scores)
    score_array = -score_array + abs(np.max(score_array))

    probabilities = score_array / np.sum(score_array)

    indices = list(range(len(population)))
    indices_array = np.array(indices)

    selected_indices = np.random.choice(
        indices_array, size=next_gen_number, p=probabilities
    )

    selected = []
    for indx in selected_indices:
        selected.append(population[indx])

    return selected


def quaternary_tournament(population, scores, next_gen_number, random_seed=42):
    """Selects next generation using quaternary tournament selection for
    single objective.

    This function implements quaternary tournament selection to select a
    specific number of members for the next generation.

    Args:
        population: A list containing members of current population with
            parents and children combined.
        scores: A list containing scores of each member of the
            population. The format is:
                [member 1 score, member 2 score, ....]
            The order of members has to be consistent with population
            argument.
        next_gen_number: An int indicating the number of members to be
            selected for the next generation.
        random_seed: An int indicating the seed of the random number
            generator.

    Returns:
        A list of members for the next generation of population.
    """

    np.random.seed(random_seed)

    indices = list(range(len(population)))
    indices_array = np.array(indices)

    selected = []
    for i in range(next_gen_number):
        best_score = math.inf
        picked = None
        selected_indices = np.random.choice(indices_array, size=4)

        for indx in selected_indices:
            if scores[indx] < best_score:
                best_score = scores[indx]
                picked = population[indx]

        selected.append(picked)

    return selected


def uniform_stochastic(population, next_gen_number, random_seed=42):
    """Selects next generation using uniform stochastic selection for
    single objective.

    This function implements uniform stochastic selection to select a
    specific number of members for the next generation.

    Args:
        population: A list containing members of current population with
            parents and children combined.
        next_gen_number: An int indicating the number of members to be
            selected for the next generation.
        random_seed: An int indicating the seed of the random number
            generator.

    Returns:
        A list of members for the next generation of population.
    """

    np.random.seed(random_seed)

    indices = list(range(len(population)))
    indices_array = np.array(indices)

    selected_indices = np.random.choice(
        indices_array, size=next_gen_number
    )

    selected = []
    for indx in selected_indices:
        selected.append(population[indx])

    return selected
