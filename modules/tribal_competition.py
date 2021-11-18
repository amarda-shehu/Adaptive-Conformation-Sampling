# tribal_competition.py
# author: Ahmed Bin Zaman
# since: 01/2019

"""Module for tribal competition on tribal EAs.

This module provides functions to perform competition between different
tribes.

Available Functions:
- free_energy_competition: Performs free energy competition between the
    tribes.
- avg_tribe_fitness: Calculates average fitness of a tribe.
- free_energy_fitness: Calculates free energy fitness of a tribe.
"""

import random


def free_energy_competition(tribes, member_fitness, temperature):
    """Performs free energy competition between the tribes.

    This function implements free energy tribal competition via
    increasing the size of the best tribe by replicating the best
    solution in that tribe once, decreasing the size of the worst tribe by
    discarding the worst solution in that tribe, and utilizing free
    energy fitness of the tribes for the competition. The lower the
    fitness score, the better the tribe.

    Args:
        tribes: A 2D list containing the tribes. The format is:
                [
                    [tribe 1 member 1, tribe 1 member 2,....],
                    [tribe 2 member 1, tribe 2 member 2,....],
                    ....
                ]
        member_fitness: A 2D list containing fitness scores of each
            member of the tribes. The format is:
                [
                    [tribe 1 member 1 score, tribe 1 member 2 score,..],
                    [tribe 2 member 1 score, tribe 2 member 2 score,..],
                    ....
                ]
            The order of tribes has to be consistent with
            tribes argument.
        temperature: A float containing the temperature for free energy
            calculation.

    Returns:
        A 2D list of tribes as a result of the competition. The format
        is:
            [
                [tribe 1 member 1, tribe 1 member 2,....],
                [tribe 2 member 1, tribe 2 member 2,....],
                ....
            ]
    """

    no_of_tribes = len(tribes)
    if no_of_tribes == 1:
        return tribes

    final_tribes = []

    best_indices = []
    worst_indices = []

    # Assess fitness of each tribe
    tribal_fitness = []
    for i in range(no_of_tribes):
        tribal_fitness.append(
            free_energy_fitness(member_fitness[i], temperature)
        )

    # Figure out best and worst tribes
    best_fitness = min(tribal_fitness)
    worst_fitness = max(tribal_fitness)

    for i in range(len(tribal_fitness)):
        if tribal_fitness[i] == best_fitness:
            best_indices.append(i)
        if tribal_fitness[i] == worst_fitness:
            worst_indices.append(i)

    best_index = -1
    worst_index = -1
    while best_index == worst_index:
        best_index = random.randint(0, len(best_indices) - 1)
        best_index = best_indices[best_index]
        worst_index = random.randint(0, len(worst_indices) - 1)
        worst_index = worst_indices[worst_index]

    # Update the tribes
    if best_index < worst_index:
        final_tribes.extend(tribes[:best_index])
        best_tribe = tribes[best_index]
        best_member = member_fitness[best_index].index(
            min(member_fitness[best_index])
        )
        best_tribe.append(best_tribe[best_member])
        final_tribes.append(best_tribe)

        final_tribes.extend(tribes[best_index + 1:worst_index])
        worst_tribe = tribes[worst_index]
        worst_member = member_fitness[worst_index].index(
            max(member_fitness[worst_index])
        )
        del worst_tribe[worst_member]
        if len(worst_tribe) != 0:
            final_tribes.append(worst_tribe)

        final_tribes.extend(tribes[worst_index + 1:])

    else:
        final_tribes.extend(tribes[:worst_index])
        worst_tribe = tribes[worst_index]
        worst_member = member_fitness[worst_index].index(
            max(member_fitness[worst_index])
        )
        del worst_tribe[worst_member]

        if len(worst_tribe) != 0:
            final_tribes.append(worst_tribe)

        final_tribes.extend(tribes[worst_index + 1:best_index])
        best_tribe = tribes[best_index]
        best_member = member_fitness[best_index].index(
            min(member_fitness[best_index])
        )
        best_tribe.append(best_tribe[best_member])
        final_tribes.append(best_tribe)

        final_tribes.extend(tribes[best_index + 1:])

    return final_tribes


def avg_tribe_fitness(tribe_scores):
    """Calculates average fitness of a tribe.

    This function implements a fitness function based on the average
    score for evaluating a tribe.

    Args:
        tribe_scores: A list containing fitness scores of each
            member of the tribe. The format is:
                [member 1 score, member 2 score, ....]

    Returns:
        A float containing the fitness score for the given tribe.
    """

    return sum(tribe_scores) / len(tribe_scores)


def free_energy_fitness(tribe_scores, temperature):
    """Calculates free energy fitness of a tribe.

    This function implements a fitness function based on the free energy
    score for evaluating a tribe.

    Args:
        tribe_scores: A list containing fitness scores of each
            member of the tribe. The format is:
                [member 1 score, member 2 score, ....]
        temperature: A float containing the temperature for free energy
            calculation.

    Returns:
        A float containing the fitness score for the given tribe.
    """

    return avg_tribe_fitness(tribe_scores) + (temperature * len(tribe_scores))
