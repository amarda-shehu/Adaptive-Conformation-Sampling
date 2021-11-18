# spea.py
# author: Ahmed Bin Zaman
# since: 01/2019

from modules import population_meta as pl
from modules import io
from modules import improvement_meta as imp
from modules import selection as st
from modules import tribal_utility as tu
from modules import tribal_competition as tc
import pyrosetta as pr
import configparser as cp
import math
import os
import random

# ***************
# Initial setup *
# ***************

pr.init()

# Read configuration file
conf = cp.ConfigParser()
conf.read("configs/spea.ini")

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
file_suffix = "spea_"

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

# ***********************
# Tribal initialization *
# ***********************

level_cutoff = int(conf['tribal']['rmsdLevelCutoff'])
competition_frequency = int(conf['tribal']['competitionFrequency'])
tribes = tu.leader_clustering(parents, level_cutoff)
temperature = float(conf['tribal']['temperature'])

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

generation_count = 1

while eval_budget > 0:
    new_tribes = []

    for tribe in tribes:
        # Generate children and perform local search
        children = []
        parents_scores = []
        children_scores = []

        for parent in tribe:
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

            if score4_energy < lowest_energy:
                lowest_energy = score4_energy
                lowest_energy_pose.assign(parent)

            pose_ca_rmsd1 = pr.rosetta.core.scoring.CA_rmsd(native_pose1,
                                                            parent)
            if pose_ca_rmsd1 < min_ca_rmsd1:
                min_ca_rmsd1 = pose_ca_rmsd1
                min_ca_rmsd_pose1.assign(parent)

            pose_ca_rmsd2 = pr.rosetta.core.scoring.CA_rmsd(native_pose2,
                                                            parent)
            if pose_ca_rmsd2 < min_ca_rmsd2:
                min_ca_rmsd2 = pose_ca_rmsd2
                min_ca_rmsd_pose2.assign(parent)

            f2.write(
                "%s %s %s\n" % (pose_ca_rmsd1, pose_ca_rmsd2, score4_energy))

            for i in range(1, no_of_residues + 1):
                for coordinate in parent.residue(i).xyz("CA"):
                    f.write("%.3f " % coordinate)
            f.write("\n")
            no_of_decoys += 1

            # Evaluation bookkeeping for children
            score4_energy = score4_function(improved_child)
            children_scores.append(score4_energy)

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

            f2.write(
                "%s %s %s\n" % (pose_ca_rmsd1, pose_ca_rmsd2, score4_energy))

            for i in range(1, no_of_residues + 1):
                for coordinate in improved_child.residue(i).xyz("CA"):
                    f.write("%.3f " % coordinate)
            f.write("\n")
            no_of_decoys += 1

        # Perform selection
        new_tribes.append(st.truncation(
            tribe, children, parents_scores, children_scores, elitism_rate
        ))

    # Tribal Competition
    if generation_count % competition_frequency == 0:
        tribe_scores = []

        for tribe in new_tribes:
            member_scores = []
            for member in tribe:
                member_scores.append(score4_function(member))

            tribe_scores.append(member_scores)

        tribes = tc.free_energy_competition(
            new_tribes, tribe_scores, temperature
        )
    else:
        tribes = new_tribes

    generation_count += 1

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
