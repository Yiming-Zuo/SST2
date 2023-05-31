from openmm.app import PDBFile, ForceField, HBonds, Simulation, PME

from openmm import (
    LangevinMiddleIntegrator,
    MonteCarloBarostat,
    Platform,
    unit,
    openmm,
)


def create_system_simulation(
    pdb_file,
    forcefield,
    dt=2 * unit.femtosecond,
    temperature=300 * unit.kelvin,
    friction=1 / unit.picoseconds,
    nonbondedMethod=PME,
    nonbondedCutoff=1 * unit.nanometers,
    constraints=HBonds,
    platform_name="CUDA",
    rigidWater=True,
    ewaldErrorTolerance=0.0005,
    hydrogenMass=3.0 * unit.amu,
):

    # Fix pdb path !!!!
    pdb = PDBFile(pdb_file)

    integrator = LangevinMiddleIntegrator(temperature, friction, dt)

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
    """Creates a minimized simulation object"""
    # prop = {'Precision':'single'}

    platform = Platform.getPlatformByName(platform_name)
    prop = {}
    if platform_name != "CPU":
        prop["Precision"] = "single"

    for i, force in enumerate(system.getForces()):
        force.setForceGroup(i)

    simulation = Simulation(topology, system, integrator, platform, prop)
    simulation.context.setPositions(position)

    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)
    print("Created simulation")

    return simulation
