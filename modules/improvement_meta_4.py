# improvement_meta.py
# author: Ahmed Bin Zaman
# since: 02/2021
"""Module for improving the fitness of a conformation.

This module provides functionalities like local search to improve the
current fitness of a given conformation. The bookkeeping is for
metamorphic proteins with four native structures.

Available Classes:
- Improvement: Encapsulates the operations to improve fitness of a
    conformation.
"""

import pyrosetta as pr
import math


class Improvement:
    """Encapsulates the operations to improve fitness of a conformation.

    Provides functionalities like local search to improve the current
    fitness of a given conformation.

    Public Attributes:
    - native_pose1: Contains the firts native pose provided in the
        constructor (pyrosetta Pose object).
    - native_pose2: Contains the firts native pose provided in the
        constructor (pyrosetta Pose object).
    - total_energy_evals: Total number of energy evaluations done in
        all the operations performed (integer).
    - last_op_energy_evals: Number of energy evaluations done in the
        last operation performed (integer).
    - min_ca_rmsd: Minimum Ca-RMSD value to the native conformation
        among all the conformations generated in all the operations
        (float)
    - last_op_min_ca_rmsd: Minimum Ca-RMSD value to the native
        conformation among all the conformations generated in the last
        operation performed (float).
    - min_ca_rmsd_pose: Conformation with minimum Ca-RMSD value to the
        native conformation among all the conformations generated in all
        the operations (pyrosetta Pose object).
    - last_op_min_ca_rmsd_pose: Conformation with minimum Ca-RMSD value
        to the native conformation among all the conformations generated
        in the last operation performed (pyrosetta Pose object).

    Available methods:
    - local_search: Performs greedy local search to improve fitness of a
        conformation.
    """

    def __init__(self, native_pose1, native_pose2, native_pose3, native_pose4):
        """Constructor

        Args:
            native_pose: A pyrosetta Pose object containing the native
                conformation. This is used for minimum Ca-RMSD
                calculation. If you don't need this calculation, or
                don't have the native conformation, just provide a
                random Pose object.
        """
        self.native_pose1 = pr.Pose()
        self.native_pose1.assign(native_pose1)
        self.native_pose2 = pr.Pose()
        self.native_pose2.assign(native_pose2)
        self.native_pose3 = pr.Pose()
        self.native_pose3.assign(native_pose3)
        self.native_pose4 = pr.Pose()
        self.native_pose4.assign(native_pose4)

        self.total_energy_evals = 0
        self.last_op_energy_evals = 0

        self.min_ca_rmsd1 = math.inf
        self.last_op_min_ca_rmsd1 = math.inf
        self.min_ca_rmsd_pose1 = pr.Pose()
        self.last_op_min_ca_rmsd_pose1 = pr.Pose()

        self.min_ca_rmsd2 = math.inf
        self.last_op_min_ca_rmsd2 = math.inf
        self.min_ca_rmsd_pose2 = pr.Pose()
        self.last_op_min_ca_rmsd_pose2 = pr.Pose()

        self.min_ca_rmsd3 = math.inf
        self.last_op_min_ca_rmsd3 = math.inf
        self.min_ca_rmsd_pose3 = pr.Pose()
        self.last_op_min_ca_rmsd_pose3 = pr.Pose()

        self.min_ca_rmsd4 = math.inf
        self.last_op_min_ca_rmsd4 = math.inf
        self.min_ca_rmsd_pose4 = pr.Pose()
        self.last_op_min_ca_rmsd_pose4 = pr.Pose()

    def local_search(self, pose, mover, score_function, successive_failures):
        """Performs greedy local search to improve fitness of a
        conformation.

        This local search performs specific moves to map a conformation
        to a nearby local minimum in the energy surface. The search is
        terminated when a specific number of moves fail to improve the
        score based on a specific fitness function.

        Args:
            pose: A pyrosetta Pose object containing initial
                conformation.
            mover: A pyrosetta Mover object derermining the moves in
                local search.
            score_function: A pyrosetta ScoreFunction object for scoring
                each move.
            successive_failures: An int indicating the threshold for
                consecutive number of failed moves in each trajectory.

        Returns:
            A pyrosetta Pose object containing the conformation with
            locally minimum fitness.
        """
        local_minima = pr.Pose()
        local_minima.assign(pose)

        new_pose = pr.Pose()
        new_pose.assign(pose)

        self.last_op_min_ca_rmsd1 = pr.rosetta.core.scoring.CA_rmsd(
            self.native_pose1, new_pose
        )
        self.last_op_min_ca_rmsd2 = pr.rosetta.core.scoring.CA_rmsd(
            self.native_pose2, new_pose
        )
        self.last_op_min_ca_rmsd3 = pr.rosetta.core.scoring.CA_rmsd(
            self.native_pose3, new_pose
        )
        self.last_op_min_ca_rmsd4 = pr.rosetta.core.scoring.CA_rmsd(
            self.native_pose4, new_pose
        )

        local_minima_score = score_function(local_minima)
        self.last_op_energy_evals = 1

        failed = 0
        # Perform greedy local search
        while failed < successive_failures:
            mover.apply(new_pose)

            pose_ca_rmsd1 = pr.rosetta.core.scoring.CA_rmsd(
                self.native_pose1, new_pose
            )
            pose_ca_rmsd2 = pr.rosetta.core.scoring.CA_rmsd(
                self.native_pose2, new_pose
            )
            pose_ca_rmsd3 = pr.rosetta.core.scoring.CA_rmsd(
                self.native_pose3, new_pose
            )
            pose_ca_rmsd4 = pr.rosetta.core.scoring.CA_rmsd(
                self.native_pose4, new_pose
            )

            if pose_ca_rmsd1 < self.last_op_min_ca_rmsd1:
                self.last_op_min_ca_rmsd1 = pose_ca_rmsd1
                self.last_op_min_ca_rmsd_pose1.assign(new_pose)

            if pose_ca_rmsd2 < self.last_op_min_ca_rmsd2:
                self.last_op_min_ca_rmsd2 = pose_ca_rmsd2
                self.last_op_min_ca_rmsd_pose2.assign(new_pose)

            if pose_ca_rmsd3 < self.last_op_min_ca_rmsd3:
                self.last_op_min_ca_rmsd3 = pose_ca_rmsd3
                self.last_op_min_ca_rmsd_pose3.assign(new_pose)

            if pose_ca_rmsd4 < self.last_op_min_ca_rmsd4:
                self.last_op_min_ca_rmsd4 = pose_ca_rmsd4
                self.last_op_min_ca_rmsd_pose4.assign(new_pose)

            current_score = score_function(new_pose)
            self.last_op_energy_evals += 1

            if current_score < local_minima_score:
                local_minima.assign(new_pose)
                local_minima_score = current_score
                failed = 0
            else:
                failed += 1

        # Bookkeeping
        self.total_energy_evals += self.last_op_energy_evals

        if self.last_op_min_ca_rmsd1 < self.min_ca_rmsd1:
            self.min_ca_rmsd1 = self.last_op_min_ca_rmsd1
            self.min_ca_rmsd_pose1.assign(self.last_op_min_ca_rmsd_pose1)

        if self.last_op_min_ca_rmsd2 < self.min_ca_rmsd2:
            self.min_ca_rmsd2 = self.last_op_min_ca_rmsd2
            self.min_ca_rmsd_pose2.assign(self.last_op_min_ca_rmsd_pose2)

        if self.last_op_min_ca_rmsd3 < self.min_ca_rmsd3:
            self.min_ca_rmsd3 = self.last_op_min_ca_rmsd3
            self.min_ca_rmsd_pose3.assign(self.last_op_min_ca_rmsd_pose3)

        if self.last_op_min_ca_rmsd4 < self.min_ca_rmsd4:
            self.min_ca_rmsd4 = self.last_op_min_ca_rmsd4
            self.min_ca_rmsd_pose4.assign(self.last_op_min_ca_rmsd_pose4)

        return local_minima
