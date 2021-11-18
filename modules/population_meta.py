# population_meta.py
# author: Ahmed Bin Zaman
# since: 02/2021

"""Module for creating and/or modifying a population.

This module provides functionalities to use two stages of monte-carlo
search as well as the variation operator (mutation) to generate
and/or modify a population. The bookkeeping is for metamorphic proteins
with two native structures.

Available Classes:
- Population: Encapsulates the operations in order to generate and/or
    modify a population.
"""

import pyrosetta as pr
import random
import math


class Population:
    """Encapsulates the operations in order to generate and/or modify
    a population.

    Provides functionalities to use two stages of monte-carlo search
    as well as variations like mutation and crossover to generate and/or
    modify a population.

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
    - monte_carlo_single: Performs Metropolis Monte Carlo (MMC) search a
        fixed number of times in a single trajectory.
    - monte_carlo_fixed: Performs Metropolis Monte Carlo (MMC) search a
        fixed number of times in different trajectories.
    - monte_carlo_failure: Performs Metropolis Monte Carlo (MMC) search
        starting with an initial population as different trajectories
        until a number of successive failures.
    - mutation_operator: Constructs a mutation operator to perform
        Molecular Fragment Replacements.
    """

    def __init__(self, native_pose1, native_pose2):
        """Constructor

        Args:
            native_pose1: A pyrosetta Pose object containing the first
                native conformation. This is used for minimum Ca-RMSD
                calculation. If you don't need this calculation, or
                don't have the native conformation, just provide a
                random Pose object.
            native_pose1: A pyrosetta Pose object containing the second
                native conformation. This is used for minimum Ca-RMSD
                calculation. If you don't need this calculation, or
                don't have the native conformation, just provide a
                random Pose object.
        """
        self.native_pose1 = pr.Pose()
        self.native_pose1.assign(native_pose1)
        self.native_pose2 = pr.Pose()
        self.native_pose2.assign(native_pose2)

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

    def monte_carlo_single(self, pose, mover, score_function, temperature,
                           successive_failures, temperature_increment,
                           fixed_moves):
        """Performs Metropolis Monte Carlo (MMC) search with a specified
        move, a fixed number of times in a single trajectory. Metropolis
        criteria temperature is increased by a specific value if a
        specific number of consecutive moves fail.

        Args:
            pose: A pyrosetta Pose object containing initial
                conformation.
            mover: A pyrosetta Mover object derermining the moves in MMC
                search.
            score_function: A pyrosetta ScoreFunction object for scoring
                each move.
            temperature: An int/float defining the temperature of MMC
                search.
            successive_failures: A positive int indicating the threshold
                for consecutive number of failed moves before
                temperature increase.
            temperature_increment: An int indicating the increase in
                temperature if a number of moves fail.
            fixed_moves: A positive int indicating the number of moves
                in the search.

        Returns:
            A pyrosetta Pose object containing the final conformation.
        """

        new_pose = pr.Pose()
        new_pose.assign(pose)
        mc = pr.MonteCarlo(new_pose, score_function, temperature)
        temp = temperature

        move = 0
        failed = 0

        # Perform MMC
        while move <= fixed_moves:
            mover.apply(new_pose)
            move += 1

            pose_ca_rmsd1 = pr.rosetta.core.scoring.CA_rmsd(
                self.native_pose1, new_pose
            )
            pose_ca_rmsd2 = pr.rosetta.core.scoring.CA_rmsd(
                self.native_pose2, new_pose
            )

            if pose_ca_rmsd1 < self.last_op_min_ca_rmsd1:
                self.last_op_min_ca_rmsd1 = pose_ca_rmsd1
                self.last_op_min_ca_rmsd_pose1.assign(new_pose)
            if pose_ca_rmsd2 < self.last_op_min_ca_rmsd2:
                self.last_op_min_ca_rmsd2 = pose_ca_rmsd2
                self.last_op_min_ca_rmsd_pose2.assign(new_pose)

            if mc.boltzmann(new_pose):
                failed = 0
                if temp != temperature:
                    temp = temperature
                    mc = pr.MonteCarlo(new_pose, score_function, temp)
            else:
                failed += 1

            # Increase temperature if a number of moves fails
            if failed == successive_failures:
                temp += temperature_increment
                mc = pr.MonteCarlo(new_pose, score_function, temp)
                failed = 0

        # Bookkeeping
        self.last_op_energy_evals = fixed_moves
        self.total_energy_evals += fixed_moves

        if self.last_op_min_ca_rmsd1 < self.min_ca_rmsd1:
            self.min_ca_rmsd1 = self.last_op_min_ca_rmsd1
            self.min_ca_rmsd_pose1.assign(self.last_op_min_ca_rmsd_pose1)

        if self.last_op_min_ca_rmsd2 < self.min_ca_rmsd2:
            self.min_ca_rmsd2 = self.last_op_min_ca_rmsd2
            self.min_ca_rmsd_pose2.assign(self.last_op_min_ca_rmsd_pose2)

        return new_pose

    def monte_carlo_fixed(self, pose, mover, score_function, temperature,
                           trajectory, fixed_moves):
        """Performs Metropolis Monte Carlo (MMC) search with a specified
        move, a fixed number of times in different trajectories. Each
        trajectory is created by performing moves from the initial
        conformation passed in the constructor.

        Args:
            pose: A pyrosetta Pose object containing initial
                conformation.
            mover: A pyrosetta Mover object derermining the moves in MMC
                search.
            score_function: A pyrosetta ScoreFunction object for scoring
                each move.
            temperature: An int/float defining the temperature of MMC
                search.
            trajectory: A positive int indicating the number of
                trajectories.
            fixed_moves: A positive int indicating the number of moves
                in each trajectory.

        Returns:
            A list containing the population generated by the MMC
            search.
        """
        population = []

        # Perform MMC on all trajectories
        for i in range(trajectory):
            new_pose = pr.Pose()
            new_pose.assign(pose)
            mc = pr.MonteCarlo(new_pose, score_function, temperature)
            trial_mover = pr.TrialMover(mover, mc)

            # Perform MMC for a fixed number of moves
            for j in range(fixed_moves):
                trial_mover.apply(new_pose)
                pose_ca_rmsd1 = pr.rosetta.core.scoring.CA_rmsd(
                    self.native_pose1, new_pose
                )
                pose_ca_rmsd2 = pr.rosetta.core.scoring.CA_rmsd(
                    self.native_pose2, new_pose
                )

                if pose_ca_rmsd1 < self.last_op_min_ca_rmsd1:
                    self.last_op_min_ca_rmsd1 = pose_ca_rmsd1
                    self.last_op_min_ca_rmsd_pose1.assign(new_pose)

                if pose_ca_rmsd2 < self.last_op_min_ca_rmsd2:
                    self.last_op_min_ca_rmsd2 = pose_ca_rmsd2
                    self.last_op_min_ca_rmsd_pose2.assign(new_pose)

            population.append(new_pose)

        # Bookkeeping
        self.last_op_energy_evals = (fixed_moves * trajectory)
        self.total_energy_evals += (fixed_moves * trajectory)

        if self.last_op_min_ca_rmsd1 < self.min_ca_rmsd1:
            self.min_ca_rmsd1 = self.last_op_min_ca_rmsd1
            self.min_ca_rmsd_pose1.assign(self.last_op_min_ca_rmsd_pose1)

        if self.last_op_min_ca_rmsd2 < self.min_ca_rmsd2:
            self.min_ca_rmsd2 = self.last_op_min_ca_rmsd2
            self.min_ca_rmsd_pose2.assign(self.last_op_min_ca_rmsd_pose2)

        return population

    def monte_carlo_failure(self, mover, score_function, temperature,
                           population_base, successive_failures):
        """Performs Metropolis Monte Carlo (MMC) search starting with an
        initial population as different trajectories. The search in each
        trajectory terminates when a specific number of consecutive
        moves fail.

        Args:
            mover: A pyrosetta Mover object derermining the moves in MMC
                search.
            score_function: A pyrosetta ScoreFunction object for scoring
                each move.
            temperature: An int/float defining the temperature of MMC
                search.
            population_base: A list containing the initial population.
            successive_failures: A positive int indicating the threshold
                for consecutive number of failed moves in each
                trajectory.

        Returns:
            A list containing the population generated by the MMC
            search.
        """
        population = []
        trajectory = len(population_base)
        p_base = list(population_base)
        self.last_op_energy_evals = 0

        # Perform MMC on all trajectories
        for i in range(trajectory):
            new_pose = pr.Pose()
            new_pose.assign(p_base.pop())
            mc = pr.MonteCarlo(new_pose, score_function, temperature)
            failed = 0

            # Perform MMC until a fixed number of failed consecutive
            # moves
            while failed < successive_failures:
                mover.apply(new_pose)

                pose_ca_rmsd1 = pr.rosetta.core.scoring.CA_rmsd(
                    self.native_pose1, new_pose
                )
                pose_ca_rmsd2 = pr.rosetta.core.scoring.CA_rmsd(
                    self.native_pose2, new_pose
                )

                if pose_ca_rmsd1 < self.last_op_min_ca_rmsd1:
                    self.last_op_min_ca_rmsd1 = pose_ca_rmsd1
                    self.last_op_min_ca_rmsd_pose1.assign(new_pose)
                if pose_ca_rmsd2 < self.last_op_min_ca_rmsd2:
                    self.last_op_min_ca_rmsd2 = pose_ca_rmsd2
                    self.last_op_min_ca_rmsd_pose2.assign(new_pose)

                if mc.boltzmann(new_pose):
                    failed = 0
                else:
                    failed += 1
                self.last_op_energy_evals += 1

            population.append(new_pose)

        # Bookkeeping
        self.total_energy_evals += self.last_op_energy_evals

        if self.last_op_min_ca_rmsd1 < self.min_ca_rmsd1:
            self.min_ca_rmsd1 = self.last_op_min_ca_rmsd1
            self.min_ca_rmsd_pose1.assign(self.last_op_min_ca_rmsd_pose1)

        if self.last_op_min_ca_rmsd2 < self.min_ca_rmsd2:
            self.min_ca_rmsd2 = self.last_op_min_ca_rmsd2
            self.min_ca_rmsd_pose2.assign(self.last_op_min_ca_rmsd_pose2)

        return population

    def mutation_operator(self, fragment_length, fragment_file):
        """Constructs a mutation operator to perform Molecular Fragment
        Replacements. The operator is a pyrosetta Mover, which can be
        used to introduce mutation on a population or a conformation.

        Args:
            fragment_length: An integer indicating the fragment length.
            fragment_file: A string defining the path of the file
                containing the fragments.

        Raises:
            ValueError: if fragment_file path is empty.

        Returns:
            A pyrosetta ClassicFragmentMover object that defines each
            move to be a Molecular Fragment Replacement.
        """
        if not fragment_file:
            raise ValueError("Fragment file path cannot be empty.")

        fragset = pr.rosetta.core.fragment.ConstantLengthFragSet(
            fragment_length)
        fragset.read_fragment_file(fragment_file)
        movemap = pr.MoveMap()
        movemap.set_bb(True)
        return pr.rosetta.protocols.simple_moves.ClassicFragmentMover(
            fragset, movemap)
