# io.py
# author: Ahmed Bin Zaman
# since: 06/2018
"""Module for the input/output operations.

Contains input/output utility functions needed for the de novo structure 
prediction.

Available functions:
- get_sequence_from_fasta: Returns the sequence of a protein from it's
    fasta file.
"""

import numpy as np


def get_sequence_from_fasta(fasta_file):
    """Returns the sequence of a protein from it's fasta file

    Args:
        fasta_file: A string containing the fasta file path with the
        file name and extension.

    Raises:
        ValueError: if fasta_file path is empty.

    Returns:
        A string containing the sequence of the protein.
    """

    if not fasta_file:
        raise ValueError("Path cannot be empty.")

    handle = open(fasta_file, 'r')
    try:
        sequence = handle.readlines()
    finally:
        handle.close()

    sequence = [line.strip() for line in sequence if not '>' in line]
    sequence = ''.join(sequence)

    return sequence
