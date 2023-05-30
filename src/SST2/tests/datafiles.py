#!/usr/bin/env python3
# coding: utf-8

"""Test data files."""

import os

PYTEST_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_FILE_PATH = os.path.join(PYTEST_DIR, "inputs/")

PDB_PROT_PEP_SOL = os.path.join(TEST_FILE_PATH, "2HPL_equi_water.pdb")
