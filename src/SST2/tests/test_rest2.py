#!/usr/bin/env python3
# coding: utf-8

"""
Tests for rest2 functions
"""

import pdb_numpy
from .datafiles import PDB_PROT_PEP_SOL

def test_peptide_protein_complex(tmp_path):
    """Test peptide protein complex"""
    
    prot_pep_coor = pdb_numpy.Coor(PDB_PROT_PEP_SOL)