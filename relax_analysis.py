# relax_analysis.py
# relax_analysis.py
# author: Ahmed Bin Zaman
# since: 02/2021

from modules import io
import pyrosetta as pr
import configparser as cp
from Bio import PDB


pr.init()

# Read configuration file
conf = cp.ConfigParser()
conf.read("configs/relax_analysis.ini")

pdb1 = conf['init']['pdb1']
pdb2 = conf['init']['pdb2']

pdb1path = conf['init']['pdbPath']
pdb2path = conf['init']['pdbPath']

pose1 = pr.pose_from_pdb(pdb1path + pdb1 + ".pdb")
pose2 = pr.pose_from_pdb(pdb2path + pdb2 + ".pdb")

pose1_ctd = pr.Pose()
pose2_ctd = pr.Pose()
pose1_ctd.assign(pose1)
pose2_ctd.assign(pose2)

switch = pr.SwitchResidueTypeSetMover("centroid")
switch.apply(pose1_ctd)
switch.apply(pose2_ctd)

# Create score functions for evaluation
score4_function = pr.create_score_function("score3", "score4L")
score4_function.set_weight(pr.rosetta.core.scoring.rama, 1)
fa_scorefxn = pr.create_score_function("score12")
relax = pr.rosetta.protocols.relax.FastRelax()
relax.set_scorefxn(fa_scorefxn)

# Evaluate
energy_pre1 = score4_function(pose1_ctd)
energy_pre2 = score4_function(pose2_ctd)

relax.apply(pose1)
relax.apply(pose2)

relaxed1_ctd = pr.Pose()
relaxed2_ctd = pr.Pose()
relaxed1_ctd.assign(pose1)
relaxed2_ctd.assign(pose2)
switch.apply(relaxed1_ctd)
switch.apply(relaxed2_ctd)

energy1_post = score4_function(relaxed1_ctd)
energy2_post = score4_function(relaxed2_ctd)

print("Target 1 Energy pre-relax: %.1f" % energy_pre1)
print("Target 2 Energy pre-relax: %.1f" % energy_pre2)
print("Target 1 Energy Post-relax: %.1f" % energy1_post)
print("Target 2 Energy Post-relax: %.1f" % energy2_post)
print("Target 1 lRMSD Pre vs. Post: %.1f" % pr.rosetta.core.scoring.CA_rmsd(pose1_ctd, relaxed1_ctd))
print("Target 2 lRMSD Pre vs. Post: %.1f" % pr.rosetta.core.scoring.CA_rmsd(pose2_ctd, relaxed2_ctd))

