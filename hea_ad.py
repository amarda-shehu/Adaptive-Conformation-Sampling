# hea_ad.py
# author: Ahmed Bin Zaman
# since: 08/2020

from modules import population_meta as pl
from modules import io
from modules import improvement_meta as imp
from modules import selection as st
import pyrosetta as pr
import configparser as cp
import math
import random
import os

# ***************
# Initial setup *
# ***************

pr.init()

# Read configuration file
conf = cp.ConfigParser()
conf.read("configs/hea_ad.ini")

# Read fasta file and generate a pose from the sequence
pose = pr.pose_from_sequence(
    io.get_sequence_from_fasta(conf['init']['fastaPath'])
)

# Native pose
native_pose1 = pr.pose_from_pdb(conf['init']['pdb1Path'])
native_pose2 = pr.pose_from_pdb(conf['init']['pdb2Path'])

# Convert the pose to coarse-grained representation
switch = pr.SwitchResidueTypeSetMover("centroid")
switch.apply(pose)
switch.apply(native_pose1)
switch.apply(native_pose2)

# Create score function for evaluation
score4_function = pr.create_score_function("score3", "score4L")
score4_function.set_weight(pr.rosetta.core.scoring.rama, 1)

population_size = int(conf['init']['population'])

eval_budget = int(conf['init']['evalbudget'])

min_ca_rmsd1 = math.inf
min_ca_rmsd2 = math.inf
lowest_energy = math.inf

min_ca_rmsd_pose1 = pr.Pose()
min_ca_rmsd_pose2 = pr.Pose()
lowest_energy_pose = pr.Pose()

# File initializations
file_prefix = conf['output']['filePrefix']
file_suffix = "hea_ad_"

no_of_residues = pose.total_residue()
no_of_coords = no_of_residues * 3

f = open(file_prefix + "coordinates/" + file_suffix
         + "native_coordinates1.txt", "a")

f.write(str(no_of_coords) + "\n")

for i in range(1, no_of_residues + 1):
    for coordinate in native_pose1.residue(i).xyz("CA"):
        f.write("%.3f " % coordinate)
f.close()

f = open(file_prefix + "coordinates/" + file_suffix
         + "native_coordinates2.txt", "a")

f.write(str(no_of_coords) + "\n")

for i in range(1, no_of_residues + 1):
    for coordinate in native_pose2.residue(i).xyz("CA"):
        f.write("%.3f " % coordinate)
f.close()

f = open(file_prefix + "coordinates/" + file_suffix + "coordinates.txt", "a")

no_of_decoys = 0

f2 = open(file_prefix + "energy_rmsd/" + file_suffix + ".txt", "a")

# *****************************
# Generate initial population *
# *****************************

pop = pl.Population(native_pose1, native_pose2)

# Create score function for stage1 MMC
scorefxn = pr.create_score_function(conf['initPopulation']['stage1score'])

# Create MFR mover for initial population
mover = pop.mutation_operator(
    int(conf['initPopulation']['fragmentLength']),
    conf['initPopulation']['fragmentFile']
)

# Run stage1 MMC
stage1_pop = pop.monte_carlo_fixed(
    pose, mover, scorefxn, int(conf['initPopulation']['stage1temperature']),
    population_size, int(conf['initPopulation']['stage1moves'])
)

# Create score function for stage2 MMC
scorefxn = pr.create_score_function(conf['initPopulation']['stage2score'])

# Run stage2 MMC
parents = pop.monte_carlo_failure(
    mover, scorefxn, float(conf['initPopulation']['stage2temperature']),
    stage1_pop, int(conf['initPopulation']['stage2successiveFailures'])
)

eval_budget -= pop.total_energy_evals

if pop.min_ca_rmsd1 < min_ca_rmsd1:
    min_ca_rmsd1 = pop.min_ca_rmsd1
    min_ca_rmsd_pose1.assign(pop.min_ca_rmsd_pose1)

    ca_rmsd2 = pr.rosetta.core.scoring.CA_rmsd(
        native_pose2, min_ca_rmsd_pose1
    )

    f2.write("%s %s %s\n" %
             (min_ca_rmsd1, ca_rmsd2, score4_function(pop.min_ca_rmsd_pose1)))

    for i in range(1, no_of_residues + 1):
        for coordinate in pop.min_ca_rmsd_pose1.residue(i).xyz("CA"):
            f.write("%.3f " % coordinate)
    f.write("\n")
    no_of_decoys += 1

if pop.min_ca_rmsd2 < min_ca_rmsd2:
    min_ca_rmsd2 = pop.min_ca_rmsd2
    min_ca_rmsd_pose2.assign(pop.min_ca_rmsd_pose2)

    ca_rmsd1 = pr.rosetta.core.scoring.CA_rmsd(
        native_pose1, min_ca_rmsd_pose2
    )

    f2.write("%s %s %s\n" %
             (ca_rmsd1, min_ca_rmsd2, score4_function(pop.min_ca_rmsd_pose2)))

    for i in range(1, no_of_residues + 1):
        for coordinate in pop.min_ca_rmsd_pose2.residue(i).xyz("CA"):
            f.write("%.3f " % coordinate)
    f.write("\n")
    no_of_decoys += 1

# ****************************
# Run evolutionary framework *
# ****************************

improvement = imp.Improvement(native_pose1, native_pose2)

# Create score function for local search
scorefxn = pr.create_score_function(conf['improvement']['score'])

# Get all parameters for local search and selection
frag_length = int(conf['improvement']['fragmentLength'])
frag_file = conf['improvement']['fragmentFile']
successive_failures = int(conf['improvement']['successiveFailures'])
elitism_rate = float(conf['selection']['elitismRate'])

# Create MFR mover for local search
mover = pop.mutation_operator(frag_length, frag_file)

generation = 0
switched = 0
checked = 0
fp = 0
qt = 0
us = 0
tr = 0

selection_operator = 1
best_so_far_global = math.inf
best_so_far_prev = math.inf
keep_decreasing = False
keep_increasing = False

while eval_budget > 0:
    generation += 1

    # Generate children and perform local search
    children = []
    parents_scores = []
    children_scores = []

    for parent in parents:
        child = pr.Pose()
        child.assign(parent)

        # Perform mutation
        mover.apply(child)

        # Perform local search
        improved_child = pr.Pose()
        improved_child.assign(improvement.local_search(
            child, mover, scorefxn, successive_failures
        ))
        children.append(improved_child)
        eval_budget -= improvement.last_op_energy_evals

        # Evaluation bookkeeping for parents
        score4_energy = score4_function(parent)
        parents_scores.append(score4_energy)

        if score4_energy < best_so_far_global:
            best_so_far_global = score4_energy

        if score4_energy < lowest_energy:
            lowest_energy = score4_energy
            lowest_energy_pose.assign(parent)

        pose_ca_rmsd1 = pr.rosetta.core.scoring.CA_rmsd(native_pose1, parent)
        if pose_ca_rmsd1 < min_ca_rmsd1:
            min_ca_rmsd1 = pose_ca_rmsd1
            min_ca_rmsd_pose1.assign(parent)

        pose_ca_rmsd2 = pr.rosetta.core.scoring.CA_rmsd(native_pose2, parent)
        if pose_ca_rmsd2 < min_ca_rmsd2:
            min_ca_rmsd2 = pose_ca_rmsd2
            min_ca_rmsd_pose2.assign(parent)

        f2.write("%s %s %s\n" % (pose_ca_rmsd1, pose_ca_rmsd2, score4_energy))

        for i in range(1, no_of_residues + 1):
            for coordinate in parent.residue(i).xyz("CA"):
                f.write("%.3f " % coordinate)
        f.write("\n")
        no_of_decoys += 1

        # Evaluation bookkeeping for children
        score4_energy = score4_function(improved_child)
        children_scores.append(score4_energy)

        if score4_energy < best_so_far_global:
            best_so_far_global = score4_energy

        if score4_energy < lowest_energy:
            lowest_energy = score4_energy
            lowest_energy_pose.assign(improved_child)

        pose_ca_rmsd1 = pr.rosetta.core.scoring.CA_rmsd(native_pose1,
                                                        improved_child)
        if pose_ca_rmsd1 < min_ca_rmsd1:
            min_ca_rmsd1 = pose_ca_rmsd1
            min_ca_rmsd_pose1.assign(improved_child)

        pose_ca_rmsd2 = pr.rosetta.core.scoring.CA_rmsd(native_pose2,
                                                       improved_child)
        if pose_ca_rmsd2 < min_ca_rmsd2:
            min_ca_rmsd2 = pose_ca_rmsd2
            min_ca_rmsd_pose2.assign(improved_child)

        f2.write("%s %s %s\n" % (pose_ca_rmsd1, pose_ca_rmsd2, score4_energy))

        for i in range(1, no_of_residues + 1):
            for coordinate in improved_child.residue(i).xyz("CA"):
                f.write("%.3f " % coordinate)
        f.write("\n")
        no_of_decoys += 1

    # Perform adaptive selection
    if generation % 15 == 0:
        checked += 1
        change = abs(best_so_far_global - best_so_far_prev)

        if (change < abs(.05 * best_so_far_prev)) and (change != 0):
            keep_decreasing = False
            keep_increasing = False
            if selection_operator < 3:
                selection_operator += 1
                switched += 1
        elif change > abs(.15 * best_so_far_prev):
            keep_decreasing = False
            keep_increasing = False
            if selection_operator > 0:
                selection_operator -= 1
                switched += 1
        elif change == 0:
            if selection_operator == 0 and keep_decreasing:
                keep_decreasing = False

            if selection_operator == 3 and keep_increasing:
                keep_increasing = False

            if keep_decreasing:
                if selection_operator > 0:
                    selection_operator -= 1
                    switched += 1
            elif keep_increasing:
                if selection_operator < 3:
                    selection_operator += 1
                    switched += 1
            elif selection_operator == 3:
                selection_operator -= 1
                switched += 1
                keep_decreasing = True
                keep_increasing = False
            elif selection_operator == 0:
                selection_operator += 1
                switched += 1
                keep_increasing = True
                keep_decreasing = False
            else:
                keep_decreasing = False
                keep_increasing = False
                if selection_operator < 3:
                    selection_operator += 1
                    switched += 1
        else:
            keep_decreasing = False
            keep_increasing = False

        best_so_far_prev = best_so_far_global

    if selection_operator == 0:
        parents = st.uniform_stochastic(
            parents + children, population_size
        )
        us += 1
    elif selection_operator == 1:
        parents = st.fitness_proportional(
            parents + children, parents_scores + children_scores,
            population_size
        )
        fp += 1
    elif selection_operator == 2:
        parents = st.quaternary_tournament(
            parents + children, parents_scores + children_scores,
            population_size
        )
        qt += 1
    elif selection_operator == 3:
        parents = st.truncation(
            parents, children, parents_scores, children_scores, elitism_rate
        )
        tr += 1

if improvement.min_ca_rmsd1 < min_ca_rmsd1:
    min_ca_rmsd1 = improvement.min_ca_rmsd1
    min_ca_rmsd_pose1.assign(improvement.min_ca_rmsd_pose1)

    ca_rmsd2 = pr.rosetta.core.scoring.CA_rmsd(
        native_pose2, min_ca_rmsd_pose1
    )

    f2.write("%s %s %s\n" % (
        min_ca_rmsd1, ca_rmsd2, score4_function(improvement.min_ca_rmsd_pose1)
    ))

    for i in range(1, no_of_residues + 1):
        for coordinate in improvement.min_ca_rmsd_pose1.residue(i).xyz("CA"):
            f.write("%.3f " % coordinate)
    f.write("\n")
    no_of_decoys += 1

if improvement.min_ca_rmsd2 < min_ca_rmsd2:
    min_ca_rmsd2 = improvement.min_ca_rmsd2
    min_ca_rmsd_pose2.assign(improvement.min_ca_rmsd_pose2)

    ca_rmsd1 = pr.rosetta.core.scoring.CA_rmsd(
        native_pose1, min_ca_rmsd_pose2
    )

    f2.write("%s %s %s\n" % (
        ca_rmsd1, min_ca_rmsd2, score4_function(improvement.min_ca_rmsd_pose2)
    ))

    for i in range(1, no_of_residues + 1):
        for coordinate in improvement.min_ca_rmsd_pose2.residue(i).xyz("CA"):
            f.write("%.3f " % coordinate)
    f.write("\n")
    no_of_decoys += 1

f.close()
f2.close()

# Write Output
f = open(file_prefix + "coordinates/" + file_suffix + "coordinates.txt", "r+")
file_data = f.read()
f.seek(0, 0)
f.write(str(no_of_coords) + " " + str(no_of_decoys) + "\n" + file_data)
f.close()

switch = pr.SwitchResidueTypeSetMover("fa_standard")
switch.apply(lowest_energy_pose)
switch.apply(min_ca_rmsd_pose1)
switch.apply(min_ca_rmsd_pose2)

min_ca_rmsd_pose1.dump_pdb(file_prefix + "poses/" + file_suffix + "rmsd1.pdb")
min_ca_rmsd_pose2.dump_pdb(file_prefix + "poses/" + file_suffix + "rmsd2.pdb")
