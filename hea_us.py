# hea_us.py
# author: Ahmed Bin Zaman
# since: 08/2020

from modules import population as pl
from modules import io
from modules import improvement as imp
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
conf.read("configs/hea_us.ini")

# Read fasta file and generate a pose from the sequence
pose = pr.pose_from_sequence(
    io.get_sequence_from_fasta(conf['init']['fastaPath'])
)

# Native pose
native_pose = pr.pose_from_pdb(conf['init']['pdbPath'])

# Convert the pose to coarse-grained representation
switch = pr.SwitchResidueTypeSetMover("centroid")
switch.apply(pose)
switch.apply(native_pose)

# Create score function for evaluation
score4_function = pr.create_score_function("score3", "score4L")
score4_function.set_weight(pr.rosetta.core.scoring.rama, 1)

population_size = int(conf['init']['population'])

eval_budget = int(conf['init']['evalbudget'])

min_ca_rmsd = math.inf
lowest_energy = math.inf

min_ca_rmsd_pose = pr.Pose()
lowest_energy_pose = pr.Pose()

# File initializations
file_prefix = conf['output']['filePrefix']
file_suffix = "hea_us_"

no_of_residues = pose.total_residue()
no_of_coords = no_of_residues * 3

f = open(file_prefix + "coordinates/" + file_suffix
         + "native_coordinates.txt", "a")

f.write(str(no_of_coords) + "\n")

for i in range(1, no_of_residues + 1):
    for coordinate in native_pose.residue(i).xyz("CA"):
        f.write("%.3f " % coordinate)
f.close()

f = open(file_prefix + "coordinates/" + file_suffix + "coordinates.txt", "a")

no_of_decoys = 0

f2 = open(file_prefix + "energy_rmsd/" + file_suffix + ".txt", "a")

# *****************************
# Generate initial population *
# *****************************

pop = pl.Population(native_pose)

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

if pop.min_ca_rmsd < min_ca_rmsd:
    min_ca_rmsd = pop.min_ca_rmsd
    min_ca_rmsd_pose.assign(pop.min_ca_rmsd_pose)
    f2.write("%s %s\n" % (min_ca_rmsd, score4_function(pop.min_ca_rmsd_pose)))

    for i in range(1, no_of_residues + 1):
        for coordinate in pop.min_ca_rmsd_pose.residue(i).xyz("CA"):
            f.write("%.3f " % coordinate)
    f.write("\n")
    no_of_decoys += 1

# ****************************
# Run evolutionary framework *
# ****************************

improvement = imp.Improvement(native_pose)

# Create score function for local search
scorefxn = pr.create_score_function(conf['improvement']['score'])

# Get all parameters for local search and selection
frag_length = int(conf['improvement']['fragmentLength'])
frag_file = conf['improvement']['fragmentFile']
successive_failures = int(conf['improvement']['successiveFailures'])

# Create MFR mover for local search
mover = pop.mutation_operator(frag_length, frag_file)

while eval_budget > 0:
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

        if score4_energy < lowest_energy:
            lowest_energy = score4_energy
            lowest_energy_pose.assign(parent)

        pose_ca_rmsd = pr.rosetta.core.scoring.CA_rmsd(native_pose, parent)
        if pose_ca_rmsd < min_ca_rmsd:
            min_ca_rmsd = pose_ca_rmsd
            min_ca_rmsd_pose.assign(parent)

        f2.write("%s %s\n" % (pose_ca_rmsd, score4_energy))

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

        pose_ca_rmsd = pr.rosetta.core.scoring.CA_rmsd(native_pose,
                                                       improved_child)
        if pose_ca_rmsd < min_ca_rmsd:
            min_ca_rmsd = pose_ca_rmsd
            min_ca_rmsd_pose.assign(improved_child)

        f2.write("%s %s\n" % (pose_ca_rmsd, score4_energy))

        for i in range(1, no_of_residues + 1):
            for coordinate in improved_child.residue(i).xyz("CA"):
                f.write("%.3f " % coordinate)
        f.write("\n")
        no_of_decoys += 1

    # Perform selection
    parents = st.uniform_stochastic(
        parents + children, population_size, seed
    )

if improvement.min_ca_rmsd < min_ca_rmsd:
    min_ca_rmsd = improvement.min_ca_rmsd
    min_ca_rmsd_pose.assign(improvement.min_ca_rmsd_pose)
    f2.write("%s %s\n" % (
        min_ca_rmsd, score4_function(improvement.min_ca_rmsd_pose)
    ))

    for i in range(1, no_of_residues + 1):
        for coordinate in improvement.min_ca_rmsd_pose.residue(i).xyz("CA"):
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
switch.apply(min_ca_rmsd_pose)

min_ca_rmsd_pose.dump_pdb(file_prefix + "poses/" + file_suffix + "rmsd.pdb")
