# tribal_utility.py
# author: Ahmed Bin Zaman
# since: 01/2019

"""Module for a utility function for tribal EAs.

This module provides a function needed for tribal EAs.

Available Functions:
- leader_clustering: Performs leader clustering on a population to
    divide the members into tribes.

"""

import pyrosetta as pr


def leader_clustering(population, distance):
    """Performs leader clustering on a population to divide the members
    into tribes.

    This function implements leader clustering in order to divide the
    members into tribes. Each member who are within a certain distance
    from another member are clustered together.

    Args:
        population: A list containing the population.
        distance: An int indicating the distance cutoff.

    Returns:
        A 2D list of tribes as a result of the clustering. The format
        is:
            [
                [tribe 1 member 1, tribe 1 member 2,....],
                [tribe 2 member 1, tribe 2 member 2,....],
                ....
            ]
    """
    leaders = []
    tribes = []

    # Perform clustering
    for member in population:
        clustered = False
        for i in range(len(leaders)):
            if pr.rosetta.core.scoring.CA_rmsd(member, leaders[i]) < distance:
                tribes[i].append(member)
                clustered = True
                break

        if not clustered:
            leaders.append(member)
            tribes.append([member])

    return tribes
