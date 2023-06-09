#!/usr/bin/env python3
# coding: utf-8

"""
Tests for rest2 functions
"""

import os
import pytest

import openmm
from openmm import unit
import openmm.app as app

import pdb_numpy

import SST2.tools as tools
import SST2.rest2 as rest2


from .datafiles import PDB_PROT_PEP_SOL


def test_peptide_protein_complex(tmp_path):
    """Test peptide protein complex"""

    tolerance = 0.00001


    # Prepare system coordinates
    prot_pep_coor = pdb_numpy.Coor(PDB_PROT_PEP_SOL)

    # Write solute and solvent coordinates
    solute_coor = prot_pep_coor.select_atoms("chain B")
    solute_coor.write(os.path.join(tmp_path, "solute.pdb"))

    solvent_coor = prot_pep_coor.select_atoms("not chain B")
    solvent_coor.write(os.path.join(tmp_path, "solvent.pdb"))

    # Set system parameters
    forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")

    dt = 2 * unit.femtosecond
    temperature = 300 * unit.kelvin
    friction = 1 / unit.picoseconds


    # Set system
    pdb = app.PDBFile(PDB_PROT_PEP_SOL)
    integrator = openmm.LangevinMiddleIntegrator(temperature, friction, dt)
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1 * unit.nanometers,
        constraints=app.HBonds,
    )
    simulation = tools.setup_simulation(system, pdb.positions, pdb.topology, integrator)

    print("System Forces:")
    tools.print_forces(system, simulation)
    forces_sys = tools.get_forces(system, simulation)

    assert 1339.07153 == pytest.approx(
        forces_sys[0]["energy"].value_in_unit(unit.kilojoule_per_mole), tolerance
    )
    assert 3484.75097 == pytest.approx(
        forces_sys[1]["energy"].value_in_unit(unit.kilojoule_per_mole), tolerance
    )
    assert -192565.3092 == pytest.approx(
        forces_sys[2]["energy"].value_in_unit(unit.kilojoule_per_mole), tolerance
    )
    assert 5415.10546 == pytest.approx(
        forces_sys[3]["energy"].value_in_unit(unit.kilojoule_per_mole), tolerance
    )
    assert 0.0 == forces_sys[4]["energy"].value_in_unit(unit.kilojoule_per_mole)
    assert -182326.38123 == pytest.approx(
        forces_sys[6]["energy"].value_in_unit(unit.kilojoule_per_mole), tolerance
    )

    # Solute system:

    pdb_pep = app.PDBFile(os.path.join(tmp_path, "solute.pdb"))

    integrator_pep = openmm.LangevinMiddleIntegrator(temperature, friction, dt)
    system_pep = forcefield.createSystem(
        pdb_pep.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1 * unit.nanometers,
        constraints=app.HBonds,
    )
    simulation_pep = tools.setup_simulation(
        system_pep, pdb_pep.positions, pdb_pep.topology, integrator_pep
    )

    print("Solute Forces:")
    tools.print_forces(system_pep, simulation_pep)
    forces_solute = tools.get_forces(system_pep, simulation_pep)

    assert 55.969520 == pytest.approx(
        forces_solute[0]["energy"].value_in_unit(unit.kilojoule_per_mole), tolerance
    )
    assert 123.60030 == pytest.approx(
        forces_solute[1]["energy"].value_in_unit(unit.kilojoule_per_mole), tolerance
    )
    assert -451.44809 == pytest.approx(
        forces_solute[2]["energy"].value_in_unit(unit.kilojoule_per_mole), tolerance
    )
    assert 200.288879 == pytest.approx(
        forces_solute[3]["energy"].value_in_unit(unit.kilojoule_per_mole), tolerance
    )
    assert 0.0 == forces_solute[4]["energy"].value_in_unit(unit.kilojoule_per_mole)
    assert -71.589387 == pytest.approx(
        forces_solute[6]["energy"].value_in_unit(unit.kilojoule_per_mole), tolerance
    )

    # Solvent system

    pdb_no_pep = app.PDBFile(os.path.join(tmp_path, "solvent.pdb"))

    integrator_no_pep = openmm.LangevinMiddleIntegrator(temperature, friction, dt)

    system_no_pep = forcefield.createSystem(
        pdb_no_pep.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1 * unit.nanometers,
        constraints=app.HBonds,
    )

    simulation_no_pep = tools.setup_simulation(
        system_no_pep, pdb_no_pep.positions, pdb_no_pep.topology, integrator_no_pep
    )

    print("Solvent Forces:")
    tools.print_forces(system_no_pep, simulation_no_pep)
    forces_solvent = tools.get_forces(system_no_pep, simulation_no_pep)

    assert 1283.102050 == pytest.approx(
        forces_solvent[0]["energy"].value_in_unit(unit.kilojoule_per_mole), tolerance
    )
    assert 3361.15087 == pytest.approx(
        forces_solvent[1]["energy"].value_in_unit(unit.kilojoule_per_mole), tolerance
    )
    assert -189514.66834 == pytest.approx(
        forces_solvent[2]["energy"].value_in_unit(unit.kilojoule_per_mole), tolerance
    )
    assert 5214.81640 == pytest.approx(
        forces_solvent[3]["energy"].value_in_unit(unit.kilojoule_per_mole), tolerance
    )
    assert 0.0 == forces_solvent[4]["energy"].value_in_unit(unit.kilojoule_per_mole)
    assert -179655.59901 == pytest.approx(
        forces_solvent[6]["energy"].value_in_unit(unit.kilojoule_per_mole), tolerance
    )

    ####################
    # ## REST2 test ####
    ####################

    # Get indices of the solute atoms.
    solute_indices = [
        int(i.index) for i in pdb.topology.atoms() if i.residue.chain.id in ["B"]
    ]

    print(f"There is {len(solute_indices)} atoms in the solute group")

    integrator_rest = openmm.LangevinMiddleIntegrator(temperature, friction, dt)

    test = rest2.REST2(system, pdb, forcefield, solute_indices, integrator_rest)

    print("REST2 forces 300K:")
    tools.print_forces(test.system, test.simulation)
    forces_rest2 = tools.get_forces(test.system, test.simulation)

    (
        E_solute_scaled,
        E_solute_not_scaled,
        E_solvent,
        solvent_solute_nb,
    ) = test.compute_all_energies()

    print("Compare not scaled energy rest2 vs. classic:\n")
    print(
        f"HarmonicBondForce    {forces_rest2[0]['energy']/forces_sys[0]['energy']:.5e}"
    )
    assert forces_rest2[0]['energy']/forces_sys[0]['energy'] == 1.0
    print(
        f"HarmonicAngleForce   {forces_rest2[1]['energy']/forces_sys[1]['energy']:.5e}"
    )
    assert forces_rest2[1]['energy']/forces_sys[1]['energy'] == 1.0

    print("Compare scaled energy:")
    torsion_force = (
        forces_rest2[4]["energy"]
        + forces_rest2[5]["energy"]
        + forces_rest2[6]["energy"]
    )
    print(f"PeriodicTorsionForce {torsion_force/forces_sys[3]['energy']:.5e}")
    assert pytest.approx(torsion_force/forces_sys[3]['energy'], 0.00001) == 1.0
    print(
        f"NonbondedForce       {forces_rest2[2]['energy']/forces_sys[2]['energy']:.5e}"
    )
    assert pytest.approx(forces_rest2[2]['energy']/forces_sys[2]['energy'], 0.00001) == 1.0
    print(
        f"Total                {forces_rest2[9]['energy']/forces_sys[6]['energy']:.5e}"
    )
    assert pytest.approx(forces_rest2[9]['energy']/forces_sys[6]['energy'], 0.00001) == 1.0

    print("\nCompare torsion energy rest2 vs. solute:\n")
    torsion_force = forces_rest2[4]["energy"] + forces_rest2[5]["energy"]
    print(f"PeriodicTorsionForce {torsion_force/forces_solute[3]['energy']:.5e}")

    assert pytest.approx(torsion_force/forces_solute[3]['energy'], 0.00001) == 1.0

    print("\nCompare torsion energy rest2 vs. solvent:\n")
    torsion_force = forces_rest2[6]["energy"]
    print(f"PeriodicTorsionForce {torsion_force/forces_solvent[3]['energy']:.5e}")
    assert pytest.approx(torsion_force/forces_solvent[3]['energy'], 0.00001) == 1.0

    print("\nCompare nonbond energy rest2 vs. solute + solvent + solvent_solute_nb:\n")
    non_bonded = (
        solvent_solute_nb + forces_solute[2]["energy"] + forces_solvent[2]["energy"]
    )
    print(f"NonbondedForce       {torsion_force/forces_solvent[3]['energy']:.5e}")
    assert pytest.approx(torsion_force/forces_solvent[3]['energy'], 0.00001) == 1.0

    solute_scaled_force = forces_rest2[4]["energy"] + forces_solute[2]["energy"]
    print(f"E_solute_scaled      {solute_scaled_force/E_solute_scaled:.5e}")
    assert pytest.approx(solute_scaled_force/E_solute_scaled, 0.00001) == 1.0

    solute_not_scaled_force = (
        forces_rest2[5]["energy"]
        + forces_solute[0]["energy"]
        + forces_solute[1]["energy"]
    )

    print(f"E_solute_not_scaled  {solute_not_scaled_force} {E_solute_not_scaled}")

    print(f"E_solute_not_scaled  {non_bonded/forces_sys[2]['energy']:.5e}")

    print(f"E_solvent            {E_solvent/forces_solvent[6]['energy']:.5e}")

    """

    scale = 0.5
    test.scale_nonbonded_torsion(scale)
    print("REST2 forces 600K:")
    forces_rest2 = rest2.print_forces(test.system, test.simulation)
    (
        E_solute_scaled,
        E_solute_not_scaled,
        E_solvent,
        solvent_solute_nb,
    ) = test.compute_all_energies()
    print(f"E_solute_scaled      {E_solute_scaled}")
    print(f"E_solute_not_scaled  {E_solute_not_scaled}")
    print(f"E_solvent            {E_solvent}")
    print(f"solvent_solute_nb    {solvent_solute_nb}")

    print("\nCompare not scaled energy rest2 vs. classic:\n")
    print(
        f"HarmonicBondForce    {forces_rest2[0]['energy']/forces_sys[0]['energy']:.5e}"
    )
    print(
        f"HarmonicAngleForce   {forces_rest2[1]['energy']/forces_sys[1]['energy']:.5e}"
    )
    print("Compare scaled energy:")
    torsion_force = (
        forces_rest2[4]["energy"] * scale
        + forces_rest2[5]["energy"]
        + forces_rest2[6]["energy"]
    )
    print(f"PeriodicTorsionForce {torsion_force/forces_sys[3]['energy']:.5e}")
    print(
        f"NonbondedForce       {forces_rest2[2]['energy']/forces_sys[2]['energy']:.5e}"
    )
    print(
        f"Total                {forces_rest2[9]['energy']/forces_sys[6]['energy']:.5e}"
    )

    print("\nCompare torsion energy rest2 vs. pep:\n")
    torsion_force = forces_rest2[4]["energy"] + forces_rest2[5]["energy"]
    print(f"PeriodicTorsionForce {torsion_force/forces_solute[3]['energy']:.5e}")

    print("\nCompare torsion energy rest2 vs. no pep:\n")
    torsion_force = forces_rest2[6]["energy"]
    print(f"PeriodicTorsionForce {torsion_force/forces_solvent[3]['energy']:.5e}")

    print("\nCompare nonbond energy rest2 vs. no pep+pep+solvent_solute_nb:\n")
    
    non_bonded = (
        solvent_solute_nb + forces_solute[2]["energy"] + forces_solvent[2]["energy"]
    )
    print(f"NonbondedForce       {non_bonded/forces_sys[2]['energy']:.5e}")

    solute_scaled_force = forces_rest2[4]["energy"] + forces_solute[2]["energy"]
    print(f"E_solute_scaled      {solute_scaled_force/E_solute_scaled:.5e}")

    solute_not_scaled_force = (
        forces_rest2[5]["energy"]
        + forces_solute[0]["energy"]
        + +forces_solute[1]["energy"]
    )
    print(f"E_solute_not_scaled  {non_bonded/forces_sys[2]['energy']:.5e}")

    print(f"E_solvent            {E_solvent/forces_solvent[6]['energy']:.5e}")
    """