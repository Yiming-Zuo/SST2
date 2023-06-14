#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import logging
import time

import numpy as np
import pandas as pd

import openmm
from openmm import unit
import openmm.app as app
import pdbfixer

# Logging
logger = logging.getLogger(__name__)


def create_linear_peptide(seq, out_pdb):
    """Creates a linear peptide

    Parameters
    ----------
    seq : str
        Sequence of the peptide
    out_pdb : str
        Path to the output pdb file

    Returns
    -------
    None

    Warnings
    --------
    This function is using pdb_manip_py, this should be replaced by
    pdb_numpy in the future.
    """
    from pdb_manip_py import pdb_manip

    # Create linear peptide:
    pep_coor = pdb_manip.Coor()

    pep_coor.make_peptide(seq, out_pdb)


def prepare_pdb(in_pdb, out_pdb, pH=7.0, overwrite=False):
    """Prepare a raw pdb file adding :
    - missing residues
    - add hydrogens at user defined pH
    - add missing atoms

    Parameters
    ----------
    in_pdb : str
        Path to the input pdb file
    out_pdb : str
        Path to the output pdb file
    pH : float
        pH of the system, default is 7.0
    overwrite : bool
        Overwrite the output file, default is False
    """

    if not overwrite and os.path.isfile(out_pdb):
        logger.info(f"File {out_pdb} exists already, skip prepare_pdb() step")
        return

    # Fix peptide structure
    fixer = pdbfixer.PDBFixer(filename=in_pdb)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH)
    app.PDBFile.writeFile(fixer.topology, fixer.positions, open(out_pdb, "w"))

    return


def implicit_sim(
    pdb_in,
    forcefield,
    time,
    out_generic_name,
    temp=300 * unit.kelvin,
    dt=2 * unit.femtoseconds,
    min_steps=10000,
    save_coor_step=10000,
    overwrite=False,
):
    """Launch an implicit simulation with following steps:
    - Energy minimization with `maxIterations=min_steps`
    - Implicit simulation


    Parameters
    ----------
    pdb_in : str
        Path to the input pdb file
    forcefield : openmm ForceField
        forcefield object
    time : float
        Simulation time in ns
    out_generic_name : str
        Generic name of the output files
    temp : unit.Quantity
        Temperature, default is 300 K
    dt : unit.Quantity
        Time step, default is 2 fs
    min_steps : int
        Number of minimization steps, default is 10000
    save_coor_step : int
        Save coordinates every save_coor_step steps, default is 10000
    overwrite : bool
        Overwrite the output file, default is False

    Returns
    -------
    None
    """

    if not overwrite and os.path.isfile(f"{out_generic_name}.pdb"):
        logger.info(
            f"File {out_generic_name}.pdb exists already, skip implicit_sim() step"
        )
        return

    pdb = app.PDBFile(pdb_in)

    system = forcefield.createSystem(
        pdb.topology, nonbondedCutoff=3 * unit.nanometer, constraints=app.HBonds
    )

    tot_steps = int(1000 * time / 0.002)

    integrator = openmm.LangevinIntegrator(temp, 1 / unit.picosecond, dt)

    simulation = app.Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)

    # Minimize
    simulation.minimizeEnergy(maxIterations=min_steps)

    # Simulation
    simulation.reporters = []

    simulation.reporters.append(
        app.DCDReporter(f"{out_generic_name}.dcd", save_coor_step)
    )

    simulation.reporters.append(
        app.StateDataReporter(
            sys.stdout,
            save_coor_step,
            step=True,
            speed=True,
            remainingTime=True,
            totalSteps=tot_steps,
        )
    )

    simulation.reporters.append(
        app.StateDataReporter(
            f"{out_generic_name}.csv",
            save_coor_step,
            time=True,
            potentialEnergy=True,
            totalEnergy=True,
            temperature=True,
        )
    )

    logger.info(f"Launchinf implicit simulation of {time:.3f} ns or {tot_steps} steps")

    simulation.step(tot_steps)

    # Save position:
    positions = simulation.context.getState(
        getVelocities=False,
        getPositions=True,
        getForces=False,
        getEnergy=False,
        getParameters=False,
        groups=-1,
    ).getPositions()

    app.PDBFile.writeFile(
        pdb.topology,
        positions[: pdb.topology.getNumAtoms()],
        open(f"{out_generic_name}.pdb", "w"),
    )


def create_water_box(
    in_pdb,
    out_pdb,
    pad,
    forcefield,
    ionicStrength=0.15 * unit.molar,
    positiveIon="Na+",
    negativeIon="Cl-",
    overwrite=False,
):
    """Add a water box around a prepared pdb file.

    Parameters
    ----------
    in_pdb : str
        Path to the input pdb file
    out_pdb : str
        Path to the output pdb file
    pad : float
        Padding around the peptide in nm
    forcefield : openmm ForceField
        forcefield object
    ionicStrength : unit.Quantity
        Ionic strength of the system, default is 0.15 M
    positiveIon : str
        Positive ion, default is Na+
    negativeIon : str
        Negative ion, default is Cl-
    overwrite : bool
        Overwrite the output file, default is False
    """

    pdb = app.PDBFile(in_pdb)

    if not overwrite and os.path.isfile(out_pdb):
        logger.info(f"File {out_pdb} exists already, skip create_water_box() step")
        return pdb

    modeller = app.Modeller(pdb.topology, pdb.positions)

    # Create Box

    boxVectors = None
    geompadding = pad * unit.nanometer
    maxSize = max(
        max((pos[i] for pos in pdb.positions)) - min((pos[i] for pos in pdb.positions))
        for i in range(3)
    )
    vectors = [
        openmm.Vec3(1, 0, 0),
        openmm.Vec3(1 / 3, 2 * unit.sqrt(2) / 3, 0),
        openmm.Vec3(-1 / 3, unit.sqrt(2) / 3, unit.sqrt(6) / 3),
    ]
    boxVectors = [(maxSize + geompadding) * v for v in vectors]
    modeller.addSolvent(
        forcefield,
        boxVectors=boxVectors,
        ionicStrength=ionicStrength,
        positiveIon=positiveIon,
        negativeIon=negativeIon,
    )

    with open(out_pdb, "w") as filout:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, filout, True)
    pdb = app.PDBFile(out_pdb)

    return pdb


def create_system_simulation(
    pdb_file_io,
    forcefield,
    dt=2 * unit.femtosecond,
    temperature=300 * unit.kelvin,
    friction=1 / unit.picoseconds,
    nonbondedMethod=app.PME,
    nonbondedCutoff=1 * unit.nanometers,
    constraints=app.HBonds,
    platform_name="CUDA",
    rigidWater=True,
    ewaldErrorTolerance=0.0005,
    hydrogenMass=1.0 * unit.amu,
):
    """Creates a system and simulation object

    Parameters
    ----------
    pdb_file : str or StringIO
        Path or StringIO of the pdb file
    forcefield : Openmm forcefield
        forcefield object
    dt : unit.Quantity
        Time step, default is 2 fs
    temperature : unit.Quantity
        Temperature, default is 300 K
    friction : unit.Quantity
        Friction coefficient, default is 1 / ps
    nonbondedMethod : nonbonded method
        Nonbonded method, default is app.PME
    nonbondedCutoff : unit.Quantity
        Nonbonded cutoff, default is 1 nm
    constraints : constraint
        Constraints, default is app.HBonds
    platform_name : str
        Platform name, default is CUDA
    rigidWater : bool
        Rigid water, default is True
    ewaldErrorTolerance : float
        Ewald error tolerance, default is 0.0005
    hydrogenMass : unit.Quantity
        Hydrogen mass, default is 1 amu

    Returns
    -------
    system : openmm.System
        System object
    simulation : openmm.app.Simulation
        Simulation object
    """

    pdb = app.PDBFile(pdb_file_io)

    integrator = openmm.LangevinMiddleIntegrator(temperature, friction, dt)

    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=nonbondedMethod,
        nonbondedCutoff=nonbondedCutoff,
        constraints=constraints,
        rigidWater=rigidWater,
        ewaldErrorTolerance=ewaldErrorTolerance,
        hydrogenMass=hydrogenMass,
    )

    simulation = setup_simulation(
        system, pdb.positions, pdb.topology, integrator, platform_name
    )

    return system, simulation


def create_sim_system(pdb, forcefield, temp=300, h_mass=1.5, base_force_group=1):
    # System Configuration

    nonbondedMethod = app.PME
    ewaldErrorTolerance = 0.0005
    constraints = app.HBonds
    rigidWater = True

    if unit.is_quantity(h_mass):
        hydrogenMass = h_mass.in_units_of(unit.amu)
    else:
        hydrogenMass = h_mass * unit.amu

    nonbondedCutoff = 1.0 * unit.nanometers

    # Integration Options

    if unit.is_quantity(temp):
        temperature = temp.value_in_unit(unit.kelvin)
    else:
        temperature = temp * unit.kelvin
    pressure = 1.0 * unit.atmospheres
    barostatInterval = 25

    # Prepare the Simulation

    topology = pdb.topology
    positions = pdb.positions

    system = forcefield.createSystem(
        topology,
        nonbondedMethod=nonbondedMethod,
        nonbondedCutoff=nonbondedCutoff,
        constraints=constraints,
        rigidWater=rigidWater,
        ewaldErrorTolerance=ewaldErrorTolerance,
        hydrogenMass=hydrogenMass,
    )
    system.addForce(openmm.MonteCarloBarostat(pressure, temperature, barostatInterval))

    for force in system.getForces():
        force.setForceGroup(base_force_group)

    return system


def minimize(simulation, out_pdb, topology, maxIterations=10000, overwrite=False):
    """Minimize the energy of a system

    Parameters
    ----------
    simulation : openmm.app.Simulation
        Simulation object
    out_pdb : str
        Path to the output pdb file
    topology : openmm.app.Topology
        Topology object
    maxIterations : int
        Maximum number of iterations, default is 10000
    overwrite : bool
        Overwrite the output file, default is False
    """

    if not overwrite and os.path.isfile(out_pdb):
        logger.info(f"File {out_pdb} exists already, skip minimize() step")
        pdb = app.PDBFile(out_pdb)
        simulation.context.setPositions(pdb.positions)

    simulation.minimizeEnergy(maxIterations=maxIterations)

    # Save position:
    positions = simulation.context.getState(
        getVelocities=False,
        getPositions=True,
        getForces=False,
        getEnergy=False,
        getParameters=False,
        groups=-1,
    ).getPositions()

    app.PDBFile.writeFile(
        topology, positions[: topology.getNumAtoms()], open(out_pdb, "w")
    )


def run_sim_check_time(
    simulation, nsteps, dt, save_checkpoint_steps=None, chekpoint_name=None
):
    """Run a simulation and check the time

    Parameters
    ----------
    simulation : openmm.app.Simulation
        Simulation object
    nsteps : int
        Number of steps
    dt : unit.Quantity
        Time step
    save_checkpoint_steps : int
        Number of steps between each checkpoint
    chekpoint_name : str
        Name of the checkpoint file

    """

    print("Timing %d steps of integration..." % nsteps)
    initial_time = time.time()
    tot_steps = nsteps

    if save_checkpoint_steps is not None:
        iter_num = int(np.ceil(nsteps / save_checkpoint_steps))
        print(nsteps, save_checkpoint_steps, iter_num)
    else:
        iter_num = 1
        save_checkpoint_steps = nsteps

    for i in range(iter_num):
        if nsteps > save_checkpoint_steps:
            simulation.step(save_checkpoint_steps)
        else:
            simulation.step(nsteps)

        simulation.saveState(chekpoint_name + f"_{i:04d}.xml")

        nsteps -= save_checkpoint_steps

    final_time = time.time()
    elapsed_time = (final_time - initial_time) * unit.seconds
    elapsed_time_val = elapsed_time.value_in_unit(unit.seconds)

    dt_val = dt.value_in_unit(unit.femtoseconds)
    tot_time_val = (tot_steps * dt).value_in_unit(unit.nanoseconds)

    perfomance = (
        (tot_steps * dt).value_in_unit(unit.nanoseconds)
    ) / elapsed_time.value_in_unit(unit.day)

    print(
        f"{int(tot_steps):d} steps of {dt_val:.1f} fs timestep"
        + f" ({tot_time_val:.1f} ns) took {elapsed_time_val:.1f}"
        + f" s : {perfomance:.1f} ns/day"
    )


def setup_simulation(system, position, topology, integrator, platform_name="CUDA"):
    """Creates a simulation object

    Parameters
    ----------
    system : openmm.System
        System object
    position : unit.Quantity
        Positions
    topology : openmm.app.Topology
        Topology object
    integrator : openmm.Integrator
        Integrator object
    platform_name : str
        Platform name, default is CUDA

    Returns
    -------
    simulation : openmm.app.Simulation
        Simulation object
    """

    platform = openmm.Platform.getPlatformByName(platform_name)
    prop = {}
    if platform_name != "CPU":
        prop["Precision"] = "single"

    for i, force in enumerate(system.getForces()):
        force.setForceGroup(i)

    simulation = app.Simulation(topology, system, integrator, platform, prop)
    simulation.context.setPositions(position)

    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)
    print("Created simulation")

    return simulation


def print_forces(system, simulation):
    """Prints the forces of the system

    Parameters
    ----------
    system : openmm.System
        System object
    simulation : openmm.app.Simulation
        Simulation object

    Returns
    -------
    None
    """

    forces_dict = get_forces(system, simulation)

    for group, force in forces_dict.items():
        print(f"{group:<3} {force['name']:<25} {force['energy']}")


def get_forces(system, simulation):
    """Returns the forces of the system

    Parameters
    ----------
    system : openmm.System
        System object
    simulation : openmm.app.Simulation
        Simulation object

    Returns
    -------
    forces_dict : dict
        Dictionary with the forces
    """
    forces_dict = {}
    tot_ener = 0 * unit.kilojoules_per_mole

    for i, force in enumerate(system.getForces()):
        state = simulation.context.getState(getEnergy=True, groups={i})
        name = force.getName()
        pot_e = state.getPotentialEnergy()
        tot_ener += pot_e
        # print(f'{force.getForceGroup():<3} {name:<25} {pot_e}')

        forces_dict[force.getForceGroup()] = {"name": name, "energy": pot_e}

    forces_dict[len(forces_dict) + 1] = {"name": "Total", "energy": tot_ener}

    return forces_dict


def compute_ladder_num(generic_name, min_temp, max_temp):
    if type(min_temp) not in [int, float]:
        min_temp = min_temp._value
    if type(max_temp) not in [int, float]:
        max_temp = max_temp._value

    logger.info(f"- Extract potential energy from {generic_name}_rest2.csv")
    df_sim = pd.read_csv(generic_name + "_rest2.csv")

    # Get part number
    part = 2
    while os.path.isfile(f"{generic_name}_part_{part}.csv"):
        df_sim_part = pd.read_csv(f"{generic_name}_part_{part}.csv")
        df_sim = (
            pd.concat([df_sim, df_sim_part], axis=0, join="outer")
            .reset_index()
            .drop(["index"], axis=1)
        )
        part += 1

    # Extract potential energy
    logger.info("- Extract potential energy")
    df_sim["Solute(kJ/mol)"] = (
        df_sim["Solute scaled(kJ/mol)"] + df_sim["Solute not scaled(kJ/mol)"]
    )
    df_sim["new_pot"] = (
        df_sim["Solute(kJ/mol)"]
        + 0.5 * (min_temp / min_temp) ** 0.5 * df_sim["Solute-Solvent(kJ/mol)"]
    )
    E_pot = df_sim["new_pot"].mean()
    logger.info(f"Average Epot = {E_pot:.2e} KJ.mol-1")
    E_pot *= 8.314462618e-3
    logger.info(f"Average Epot = {E_pot:.2e} Kb")

    N_Nadler = 1 + 0.594 * np.sqrt(-E_pot) * np.log(max_temp / min_temp)
    logger.info(f"Nadler and Hansmann N = {N_Nadler:.2f}")
    N_Denshlag = 1 + (np.sqrt(-E_pot) / (2 * 0.534) - 0.5) * np.log(max_temp / min_temp)
    logger.info(f"Denshlag et al. N = {N_Denshlag:.2f}")
    N_Denshlag_2 = 1 + (0.594 * np.sqrt(-E_pot) - 1 / 2) * np.log(max_temp / min_temp)
    logger.info(f"Denshlag et al. 2 N = {N_Denshlag_2:.2f}")

    # print(f'\nHere N = {len(temp_list):.2f}')
    logger.info(f"\nHere N = {np.ceil(2*N_Denshlag):.2f}")

    return int(np.ceil(N_Denshlag))
