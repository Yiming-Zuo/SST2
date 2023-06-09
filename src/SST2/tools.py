#!/usr/bin/env python3
# coding: utf-8

import openmm
from openmm import unit
import openmm.app as app

def create_system_simulation(
    pdb_file,
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
    pdb_file : str
        Path to the pdb file
    forcefield : str
        Path to the forcefield file
    dt : unit.Quantity
        Time step, default is 2 fs
    temperature : unit.Quantity
        Temperature, default is 300 K
    friction : unit.Quantity
        Friction coefficient, default is 1 / ps
    nonbondedMethod : openmm.app.forcefield.ForceField
        Nonbonded method, default is PME
    nonbondedCutoff : unit.Quantity
        Nonbonded cutoff, default is 1 nm
    constraints : openmm.app.forcefield.ForceField
        Constraints, default is HBonds
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

    pdb = app.PDBFile(pdb_file)

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