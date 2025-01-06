#!/usr/bin/env python3
# coding: utf-8


from io import StringIO
import numpy as np
import pandas as pd
import sys
import os
import logging
import pdb_numpy.format

import openmm
from openmm import unit
import openmm.app as app

from .tools import setup_simulation, create_system_simulation, get_forces, simulate

# Logging
logger = logging.getLogger(__name__)


class Rest1Reporter(object):
    """Reporter for REST1 simulation

    Attributes
    ----------
    file : string
        The file to write to
    reportInterval : int
        The interval (in time steps) at which to write frames
    rest1 : REST1
        The REST1 object to generate the report

    Methods
    -------
    describeNextReport(simulation)
        Generate a report.


    """

    def __init__(self, file, reportInterval, rest1):
        self._out = open(file, "w", buffering=1)
        self._out.write(
            "Step,Lambda,Solute(kJ/mol),Solvent(kJ/mol),Solute-Solvent(kJ/mol)\n"
        )
        self._reportInterval = reportInterval
        self._rest1 = rest1

    def __del__(self):
        self._out.close()

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, False, False, False, False, None)

    def report(self, simulation, state):
        """Generate a report.
        Compute the energies of:
        - the solute
        - the solvent scaled
        - the solute-solvent

        Then write them to the file (`self._out`).

        Parameters
        ----------
        state : State
            The current state of the simulation

        Returns
        -------
        None

        """

        energies = self._rest1.compute_all_energies()

        # E_solute, E_solvent, solvent_solute_nb

        step = state.getStepCount()
        self._out.write(
            f"{step},{self._rest1.scale:.3f},"
            f"{energies[0].value_in_unit(unit.kilojoule_per_mole):.2f},"
            f"{energies[1].value_in_unit(unit.kilojoule_per_mole):.2f},"
            f"{energies[2].value_in_unit(unit.kilojoule_per_mole):.2f}\n"
        )


class REST1:
    """REST1 class

    Attributes
    ----------
    system : System
        The system to simulate
    simulation : Simulation
        The simulation object
    positions : coordinates
        The coordinates of the system
    topology : Topology
        The topology of the system
    solute_index : list
        The list of the solute index
    solvent_index : list
        The list of the solvent index
    system_forces : dict
        The dict of the system forces
    scale : float
        The scaling factor or lambda, default is 1.0

    init_nb_param : list
        The list of the initial nonbonded parameters (charge, sigma, epsilon)
    init_nb_exept_index : list
        The list of the exception indexes
    init_nb_exept_value : list
        The list of the initial nonbonded exception parameters
        (atom1, atom2, chargeProd, sigma, epsilon)

    solute_torsion_force : CustomTorsionForce
        The torsion force of the solute
    init_torsions_index : list
        The list of the torsion indexes
    init_torsions_value : list
        The list of the initial torsion parameters

    system_solute : Solute System
        The solute system
    simulation_solute : Solute Simulation
        The solute simulation
    system_forces_solute : Solute Forces
        The solute forces

    system_solvent : Solvent System
        The solvent system
    simulation_solvent : Solvent Simulation
        The solvent simulation
    system_forces_solvent : Solvent Forces
        The solvent forces

    init_nb_exept_solute_value : list
        The list of the initial nonbonded exception parameters of the solute
        (iatom, jatom, chargeprod, sigma, epsilon)

    Methods
    -------
    compute_all_energies()
        Compute the energies of the solute and solvent
    compute_solute_energies()
        Compute the energies of the solute
    """

    def __init__(
        self,
        system,
        pdb,
        forcefield,
        solute_index,
        integrator,
        platform_name="CUDA",
        temperature=300 * unit.kelvin,
        pressure=1.0 * unit.atmospheres,
        barostatInterval=25,
        dt=2 * unit.femtosecond,
        friction=1 / unit.picoseconds,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1 * unit.nanometers,
        constraints=app.HBonds,
        rigidWater=True,
        ewaldErrorTolerance=0.0005,
        hydrogenMass=1.0 * unit.amu,
        exclude_Pro_omegas=False,
    ):
        """Initialize the REST1 class

        To initialize the REST1 class, the following steps are performed:
        - Extract solute nonbonded index and values
        - Separate solute's torsions from the solvent
        - Extract solute's torsions index and values
        - Create separate solute and solvent simulations
        - Extract solute's nonbonded index and values from the solute_only system
        - Setup simulation

        Parameters
        ----------
        system : System
            The system to simulate
        pdb : PDBFile
            The pdb file of the system
        forcefield : ForceField
            The forcefield of the system
        solute_index : list
            The list of the solute index
        integrator : Integrator
            The integrator of the system
        platform_name : str
            The name of the platform, default is "CUDA"
        temperature : float
            The temperature of the system, default is 300 K
        pressure : float
            The pressure of the system, default is 1 atm
        barostatInterval : int
            The interval of the barostat, default is 25
        dt : float
            The timestep of the system, default is 2 fs
        friction : float
            The friction of the system, default is 1 ps-1
        nonbondedMethod : str
            The nonbonded method of the system, default is PME
        nonbondedCutoff : float
            The nonbonded cutoff of the system, default is 1 nm
        constraints : str
            The constraints of the system, default is HBonds
        rigidWater : bool
            The rigid water of the system, default is True
        ewaldErrorTolerance : float
            The Ewald error tolerance of the system, default is 0.0005
        hydrogenMass : float
            The hydrogen mass of the system, default is 1 amu
        exclude_Pro_omegas : bool
            The exclusion of the proline omegas scaling, default is False
        """

        self.system = system
        self.positions = pdb.positions
        self.topology = pdb.topology
        self.solute_index = solute_index
        self.solvent_index = list(
            set(range(self.system.getNumParticles())).difference(set(self.solute_index))
        )

        assert (
            len(self.solute_index) + len(self.solvent_index)
            == self.system.getNumParticles()
        )
        assert len(self.solute_index) != 0
        assert len(self.solvent_index) != 0

        self.system_forces = {
            type(force).__name__: force for force in self.system.getForces()
        }
        self.scale = 1.0

        # Extract solvent nonbonded index and values
        self.find_solvent_nb_index()
        # Separate Harmonic bond force from the solute
        self.separate_harmonic_bond_pot()
        # Separate Harmonic angle force from the solute
        self.separate_harmonic_angle_pot()
        # Separate solvent torsion from the solute
        self.separate_torsion_pot()
        # Create separate solute and solvent simulation
        self.create_solute_solvent_simulation(
            forcefield=forcefield,
            platform_name=platform_name,
            nonbondedMethod=nonbondedMethod,
            nonbondedCutoff=nonbondedCutoff,
            constraints=constraints,
            rigidWater=rigidWater,
            ewaldErrorTolerance=ewaldErrorTolerance,
            hydrogenMass=hydrogenMass,
            friction=friction,
            dt=dt,
        )
        # Extract solvent nonbonded index and values from the solvent_only system
        self.find_nb_solvent_system()
        self.setup_simulation(
            integrator,
            temperature=temperature,
            pressure=pressure,
            barostatInterval=barostatInterval,
            platform_name=platform_name,
        )

    def find_solvent_nb_index(self):
        """Extract initial solvant nonbonded indexes and values (charge, sigma, epsilon).
        Extract also exclusion indexes and values (chargeprod, sigma, epsilon)

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        nonbonded_force = self.system_forces["NonbondedForce"]

        # Copy particles
        self.init_nb_param = []
        for particle_index in range(nonbonded_force.getNumParticles()):
            [charge, sigma, epsilon] = nonbonded_force.getParticleParameters(
                particle_index
            )
            self.init_nb_param.append([charge, sigma, epsilon])

        # Copy solvent-solvent exclusions
        self.init_nb_exept_index = []
        self.init_nb_exept_value = []

        for exception_index in range(nonbonded_force.getNumExceptions()):
            [
                iatom,
                jatom,
                chargeprod,
                sigma,
                epsilon,
            ] = nonbonded_force.getExceptionParameters(exception_index)

            if iatom in self.solvent_index and jatom in self.solvent_index:
                self.init_nb_exept_index.append(exception_index)
                self.init_nb_exept_value.append(
                    [iatom, jatom, chargeprod, sigma, epsilon]
                )


    def separate_harmonic_bond_pot(self):
        """Use in the REST1 case as it avoid to modify
        twice the torsion terms in the rest1 system and
        in the solute system.

        Harmonic bond potential is separate in two groups:
        - the solute (not scaled one)
        - the solvent (scaled one)

        The original harmonic bond potential is deleted.


        Returns
        -------
        None
        """

        energy_expression = "0.5*k*(r-r_0)^2;"

        #    r_0 = 10.5 * unit.nanometers
        #    k = 1000 * unit.kilojoules_per_mole / unit.nanometers ** 2
        #    bond_restraint.addBond(1, 100, [k, r_0])
        #    system.addForce(bond_restraint)

        # Create the Solvent bond and not scaled solute torsion
        solvent_harmonic_bond_force = openmm.CustomBondForce(energy_expression)
        solvent_harmonic_bond_force.addPerBondParameter("r_0")
        solvent_harmonic_bond_force.addPerBondParameter("k")

        # Create the Solute bond
        solute_harmonic_bond_force = openmm.CustomBondForce(energy_expression)
        solute_harmonic_bond_force.addPerBondParameter("r_0")
        solute_harmonic_bond_force.addPerBondParameter("k")

        original_harmonic_bond_force = self.system_forces["HarmonicBondForce"]

        #bond_idxs = [sorted([i.index, j.index]) for i, j in self.topology.bonds()]

        # Store the original torsion parameters
        harmonic_bond_index = 0
        self.init_harmonic_bond_index = []
        self.init_harmonic_bond_value = []


        for i in range(original_harmonic_bond_force.getNumBonds()):
            (
                p1,
                p2,
                r_0,
                k,
            ) = original_harmonic_bond_force.getBondParameters(i)

            #print(p1, p2, r_0, k)
            #not_improper = (
            #    sorted([p1, p2]) in bond_idxs
            #    and sorted([p2, p3]) in bond_idxs
            #    and sorted([p3, p4]) in bond_idxs
            #)

            solute_in = (
                p1 in self.solute_index
                and p2 in self.solute_index
            )

            solvent_in = (
                p1 in self.solvent_index
                and p2 in self.solvent_index
            )


            if solvent_in:
                solvent_harmonic_bond_force.addBond(
                    p1, p2, [r_0, k]
                )
                self.init_harmonic_bond_index.append(harmonic_bond_index)
                self.init_harmonic_bond_value.append([p1, p2, r_0, k])
                harmonic_bond_index += 1
            elif solute_in:
                solute_harmonic_bond_force.addBond(
                    p1, p2, [r_0, k]
                )
            else:
                raise ValueError("Harmonic Bond not in solute or solvent")

        self.harmonic_bond_force = solvent_harmonic_bond_force

        logger.info("- Add new Harmonic Bond Forces")
        self.system.addForce(solute_harmonic_bond_force)
        self.system.addForce(solvent_harmonic_bond_force)

        logger.info("- Delete original Harmonic Bond Forces")

        for count, force in enumerate(self.system.getForces()):
            if isinstance(force, openmm.HarmonicBondForce):
                self.system.removeForce(count)


    def separate_harmonic_angle_pot(self):
        """Use in the REST1 case as it avoid to modify
        twice the torsion terms in the rest1 system and
        in the solute system.

        Harmonic bond potential is separate in two groups:
        - the solute (not scaled one)
        - the solvent (scaled one)

        The original harmonic bond potential is deleted.


        Returns
        -------
        None
        """

        energy_expression = "0.5*k*(theta-theta_0)^2;"

        #    r_0 = 10.5 * unit.nanometers
        #    k = 1000 * unit.kilojoules_per_mole / unit.nanometers ** 2
        #    bond_restraint.addBond(1, 100, [k, r_0])
        #    system.addForce(bond_restraint)

        # Create the Solvent bond and not scaled solute torsion
        solvent_harmonic_angle_force = openmm.CustomAngleForce(energy_expression)
        solvent_harmonic_angle_force.addPerAngleParameter("theta_0")
        solvent_harmonic_angle_force.addPerAngleParameter("k")

        # Create the Solute bond
        solute_harmonic_angle_force = openmm.CustomAngleForce(energy_expression)
        solute_harmonic_angle_force.addPerAngleParameter("theta_0")
        solute_harmonic_angle_force.addPerAngleParameter("k")

        original_harmonic_angle_force = self.system_forces["HarmonicAngleForce"]

        #bond_idxs = [sorted([i.index, j.index]) for i, j in self.topology.bonds()]

        # Store the original torsion parameters
        harmonic_angle_index = 0
        self.init_harmonic_angle_index = []
        self.init_harmonic_angle_value = []


        for i in range(original_harmonic_angle_force.getNumAngles()):
            (
                p1,
                p2,
                p3,
                theta_0,
                k,
            ) = original_harmonic_angle_force.getAngleParameters(i)

            # print(p1, p2, p3, theta_0, k)
            #not_improper = (
            #    sorted([p1, p2]) in bond_idxs
            #    and sorted([p2, p3]) in bond_idxs
            #    and sorted([p3, p4]) in bond_idxs
            #)

            solute_in = (
                p1 in self.solute_index
                and p2 in self.solute_index
                and p3 in self.solute_index
            )

            solvent_in = (
                p1 in self.solvent_index
                and p2 in self.solvent_index
                and p3 in self.solvent_index
            )


            if solvent_in:
                solvent_harmonic_angle_force.addAngle(
                    p1, p2, p3, [theta_0, k]
                )
                self.init_harmonic_angle_index.append(harmonic_angle_index)
                self.init_harmonic_angle_value.append([p1, p2, p3, theta_0, k])
                harmonic_angle_index += 1
            elif solute_in:
                solute_harmonic_angle_force.addAngle(
                    p1, p2, p3, [theta_0, k]
                )
            else:
                raise ValueError("Harmonic Bond not in solute or solvent")

        self.harmonic_angle_force = solvent_harmonic_angle_force

        logger.info("- Add new Harmonic Bond Forces")
        self.system.addForce(solute_harmonic_angle_force)
        self.system.addForce(solvent_harmonic_angle_force)

        logger.info("- Delete original Harmonic Bond Forces")

        for count, force in enumerate(self.system.getForces()):
            if isinstance(force, openmm.HarmonicAngleForce):
                self.system.removeForce(count)

    def separate_torsion_pot(self):
        """Use in the REST1 case as it avoid to modify
        twice the torsion terms in the rest1 system and
        in the solute system.

        Torsion potential is separate in two groups:
        - the solute (scaled one)
        - the solvent and not scaled solute torsion.

        As improper angles are not supposed to be scaled, here we extract only
        the proper torsion angles.

        To identify proper angles we use a trick from:
        https://github.com/maccallumlab/meld/blob/master/meld/runner/transform/rest2.py

        The original torsion potential is deleted.

        Returns
        -------
        None
        """

        energy_expression = "k*(1+cos(period*theta-phase));"

        # Create the Solvent bond and not scaled solute torsion
        solvent_torsion_force = openmm.CustomTorsionForce(energy_expression)
        solvent_torsion_force.addPerTorsionParameter("period")
        solvent_torsion_force.addPerTorsionParameter("phase")
        solvent_torsion_force.addPerTorsionParameter("k")

        # Create the Solute bond
        solute_torsion_force = openmm.CustomTorsionForce(energy_expression)
        solute_torsion_force.addPerTorsionParameter("period")
        solute_torsion_force.addPerTorsionParameter("phase")
        solute_torsion_force.addPerTorsionParameter("k")

        ## Create the not scaled Solute bond
        #solute_not_scaled_torsion_force = openmm.CustomTorsionForce(energy_expression)
        #solute_not_scaled_torsion_force.addPerTorsionParameter("period")
        #solute_not_scaled_torsion_force.addPerTorsionParameter("phase")
        #solute_not_scaled_torsion_force.addPerTorsionParameter("k")

        original_torsion_force = self.system_forces["PeriodicTorsionForce"]

        #bond_idxs = [sorted([i.index, j.index]) for i, j in self.topology.bonds()]

        # Store the original torsion parameters
        torsion_index = 0
        self.init_torsions_index = []
        self.init_torsions_value = []


        for i in range(original_torsion_force.getNumTorsions()):
            (
                p1,
                p2,
                p3,
                p4,
                periodicity,
                phase,
                k,
            ) = original_torsion_force.getTorsionParameters(i)

            #not_improper = (
            #    sorted([p1, p2]) in bond_idxs
            #    and sorted([p2, p3]) in bond_idxs
            #    and sorted([p3, p4]) in bond_idxs
            #)

            solute_in = (
                p1 in self.solute_index
                and p2 in self.solute_index
                and p3 in self.solute_index
                and p4 in self.solute_index
            )

            solvent_in = (
                p1 in self.solvent_index
                and p2 in self.solvent_index
                and p3 in self.solvent_index
                and p4 in self.solvent_index
            )


            if solvent_in:
                solvent_torsion_force.addTorsion(
                    p1, p2, p3, p4, [periodicity, phase, k]
                )
                self.init_torsions_index.append(torsion_index)
                self.init_torsions_value.append([p1, p2, p3, p4, periodicity, phase, k])
                torsion_index += 1
            elif solute_in:
                solute_torsion_force.addTorsion(
                    p1, p2, p3, p4, [periodicity, phase, k]
                )
            else:
                raise ValueError("Torsion not in solute or solvent")

        self.solvent_torsion_force = solvent_torsion_force

        logger.info("- Add new Torsion Forces")
        self.system.addForce(solute_torsion_force)
        self.system.addForce(solvent_torsion_force)

        logger.info("- Delete original Torsion Forces")

        for count, force in enumerate(self.system.getForces()):
            if isinstance(force, openmm.PeriodicTorsionForce):
                self.system.removeForce(count)

    def create_solute_solvent_simulation(
        self,
        forcefield,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1 * unit.nanometers,
        constraints=app.HBonds,
        platform_name="CUDA",
        rigidWater=True,
        ewaldErrorTolerance=0.0005,
        hydrogenMass=1.0 * unit.amu,
        friction=1 / unit.picoseconds,
        dt=2 * unit.femtosecond,
    ):
        """Extract solute only and solvent only coordinates.
        A sytem and a simulation is then created for both systems.

        Parameters
        ----------
        forcefield : str
            Forcefield name
        nonbondedMethod : Nonbonded Method
            Nonbonded method, default is app.PME
        nonbondedCutoff : float * unit.nanometers
            Nonbonded cutoff
        constraints : Constraints
            Constraints
        platform_name : str
            Platform name, default is CUDA
        rigidWater : bool
            Rigid water, default is True
        ewaldErrorTolerance : float
            Ewald error tolerance, default is 0.0005
        hydrogenMass : float * unit.amu
            Hydrogen mass, default is 1.0 * unit.amu
        friction : float / unit.picoseconds
            Friction, default is 1 / unit.picoseconds
        dt : float * unit.femtosecond
            Time step, default is 2 * unit.femtosecond

        """

        # Save pdb coordinates to read them with pdb_numpy

        # Redirect stdout in the variable new_stdout:
        old_stdout = sys.stdout
        stdout = new_stdout = StringIO()
        # In case of dummy atoms (position restraints, ...)
        # It has to be removed from pdb files
        top_num_atom = self.topology.getNumAtoms()

        app.PDBFile.writeFile(
            self.topology, self.positions[:top_num_atom], stdout, True
        )
        sys.stdout = old_stdout

        # Read
        solute_solvent_coor = pdb_numpy.format.pdb.parse(
            new_stdout.getvalue().split("\n")
        )

        # Separate coordinates in two pdb files:
        solute_coor = solute_solvent_coor.select_index(self.solute_index)
        # solute_coor.write(solute_out_pdb, overwrite=True)
        # To avoid saving a temporary file, we use StringIO
        solute_out_pdb = StringIO(pdb_numpy.format.pdb.get_pdb_string(solute_coor))

        solvent_coor = solute_solvent_coor.select_index(self.solvent_index)
        # solvent_coor.write(solvent_out_pdb, overwrite=True)
        # To avoid saving a temporary file, we use StringIO
        solvent_out_pdb = StringIO(pdb_numpy.format.pdb.get_pdb_string(solvent_coor))

        # Create system and simulations:
        self.system_solute, self.simulation_solute = create_system_simulation(
            solute_out_pdb,
            cif_format=False,
            forcefield=forcefield,
            nonbondedMethod=nonbondedMethod,
            nonbondedCutoff=nonbondedCutoff,
            constraints=constraints,
            platform_name=platform_name,
            rigidWater=rigidWater,
            ewaldErrorTolerance=ewaldErrorTolerance,
            hydrogenMass=hydrogenMass,
            friction=friction,
            dt=dt,
        )
        self.system_forces_solute = {
            type(force).__name__: force for force in self.system_solute.getForces()
        }

        self.system_solvent, self.simulation_solvent = create_system_simulation(
            solvent_out_pdb,
            cif_format=False,
            forcefield=forcefield,
            nonbondedMethod=nonbondedMethod,
            nonbondedCutoff=nonbondedCutoff,
            constraints=constraints,
            platform_name=platform_name,
            rigidWater=rigidWater,
            ewaldErrorTolerance=ewaldErrorTolerance,
            hydrogenMass=hydrogenMass,
        )

        self.system_forces_solvent = {
            type(force).__name__: force for force in self.system_solvent.getForces()
        }

    def find_nb_solvent_system(self):
        """Extract in the solvent only system:
        - exeption indexes and values (chargeprod, sigma, epsilon)

        Solvent nonbonded values are not extracted as they are identical to
        the main system. Indexes are [0 :len(nonbonded values)]

        Exception values are stored as indexes [iatom, jatom] are different.
        """

        nonbonded_force = self.system_forces_solvent["NonbondedForce"]

        # Copy particles
        self.init_nb_exept_solvent_value = []
        for exception_index in range(nonbonded_force.getNumExceptions()):
            [
                iatom,
                jatom,
                chargeprod,
                sigma,
                epsilon,
            ] = nonbonded_force.getExceptionParameters(exception_index)
            self.init_nb_exept_solvent_value.append(
                [iatom, jatom, chargeprod, sigma, epsilon]
            )

    def setup_simulation(
        self,
        integrator,
        temperature=300 * unit.kelvin,
        pressure=1.0 * unit.atmospheres,
        barostatInterval=25,
        platform_name="CUDA",
    ):
        """Add the simulation object.

        parameters
        ----------
        integrator : openmm.Integrator
            Integrator
        temperature : float * unit.kelvin
            Temperature, default is 300 * unit.kelvin
        pressure : float * unit.atmospheres
            Pressure, default is 1.0 * unit.atmospheres
        barostatInterval : int
            Barostat interval, default is 25
        platform_name : str
            Platform name, default is "CUDA"

        """

        # Add PT MonteCarlo barostat
        self.system.addForce(
            openmm.MonteCarloBarostat(pressure, temperature, barostatInterval)
        )

        self.simulation = setup_simulation(
            self.system,
            self.positions,
            self.topology,
            integrator=integrator,
            platform_name=platform_name,
        )

    def compute_solute_solvent_system_energy(self):
        """Update solute only and solvent only systems
        coordinates and box vector according to the solute-solvent
        system values.
        Extract then forces for each systems.

        Returns
        -------
        forces_solute : list of float * unit.kilojoules_per_mole / unit.nanometers
            Forces on solute
        forces_solvent : list of float * unit.kilojoules_per_mole / unit.nanometers
            Forces on solvent
        """

        sim_state = self.simulation.context.getState(getPositions=True, getEnergy=True)

        tot_positions = sim_state.getPositions(asNumpy=True)
        box_vector = sim_state.getPeriodicBoxVectors()

        self.simulation_solute.context.setPeriodicBoxVectors(*box_vector)
        self.simulation_solute.context.setPositions(tot_positions[self.solute_index])

        forces_solute = get_forces(self.system_solute, self.simulation_solute)

        self.simulation_solvent.context.setPeriodicBoxVectors(*box_vector)
        self.simulation_solvent.context.setPositions(tot_positions[self.solvent_index])

        forces_solvent = get_forces(self.system_solvent, self.simulation_solvent)

        return (forces_solute, forces_solvent)

    def update_torsions(self, scale):
        """Scale system solute torsion by a scale factor."""

        torsion_force = self.solvent_torsion_force

        for i, index in enumerate(self.init_torsions_index):
            p1, p2, p3, p4, periodicity, phase, k = self.init_torsions_value[i]
            torsion_force.setTorsionParameters(
                index, p1, p2, p3, p4, [periodicity, phase, k * scale]
            )

        torsion_force.updateParametersInContext(self.simulation.context)

    def update_nonbonded(self, scale):
        """Scale system nonbonded interaction:
        - LJ epsilon by `scale`
        - Coulomb charges by `sqrt(scale)`
        - charge product is scaled by `scale`
        """

        nonbonded_force = self.system_forces["NonbondedForce"]

        for i in self.solvent_index:
            q, sigma, eps = self.init_nb_param[i]
            nonbonded_force.setParticleParameters(
                i, q * np.sqrt(scale), sigma, eps * scale
            )

        for i in range(len(self.init_nb_exept_index)):
            index = self.init_nb_exept_index[i]
            p1, p2, q, sigma, eps = self.init_nb_exept_value[i]
            # In ExceptionParameters, q is the charge product.
            # To scale particle charges by `np.sqrt(scale)`
            # is equivalent to scale the product by `scale`
            # As for eps, eps(i,j) = sqrt(eps(i)*eps(j))
            # if we scale eps(i) and eps(j) by `scale`
            # we aslo scale eps(i,j) by `scale`.
            nonbonded_force.setExceptionParameters(
                index, p1, p2, q * scale, sigma, eps * scale
            )

        # Need to fix simulation
        nonbonded_force.updateParametersInContext(self.simulation.context)

    def update_nonbonded_solvent(self, scale):
        """Scale solvent only system nonbonded interaction:
        - LJ epsilon by `scale`
        - Coulomb charges by `sqrt(scale)`
        - charge product is scaled by `scale`
        """

        nonbonded_force = self.system_forces_solvent["NonbondedForce"]

        # assert len(self.init_nb_param) == nonbonded_force.getNumParticles()

        # for i in range(nonbonded_force.getNumParticles()):
        for i, index in enumerate(self.solvent_index):
            q, sigma, eps = self.init_nb_param[index]
            # To check we are looking at the right particles
            # q_old, sigma_old, eps_old = nonbonded_force.getParticleParameters(i)
            # if i < 4:
            #     print(f"{index}  {q}, {sigma}, {eps}")
            #     print(f"{i}  {q_old}, {sigma_old}, {eps_old}")
            nonbonded_force.setParticleParameters(
                i, q * np.sqrt(scale), sigma, eps * scale
            )
        # for particle_index in range(4):
        #    [charge, sigma, epsilon] = nonbonded_force.getParticleParameters(
        #        particle_index
        #    )
        #    print(f"{particle_index}  {charge}, {sigma}, {epsilon}")

        for i in range(nonbonded_force.getNumExceptions()):
            p1, p2, q, sigma, eps = self.init_nb_exept_solvent_value[i]
            # if i in [11, 12, 13, 14, 15, 16]:
            #    print(i, p1, p2, q, sigma, eps)
            # In ExceptionParameters, q is the charge product.
            # To scale particle charges by `np.sqrt(scale)`
            # is equivalent to scale the product by `scale`
            # As for eps, eps(i,j) = sqrt(eps(i)*eps(j))
            # if we scale eps(i) and eps(j) by `scale`
            # we aslo scale eps(i,j) by `scale`.
            nonbonded_force.setExceptionParameters(
                i, p1, p2, q * scale, sigma, eps * scale
            )

        # for exception_index in [11, 12, 13, 14, 15, 16]:
        #    [
        #        iatom,
        #        jatom,
        #        chargeprod,
        #        sigma,
        #        epsilon,
        #    ] = nonbonded_force.getExceptionParameters(exception_index)
        #    print(exception_index, iatom, jatom, chargeprod, sigma, epsilon)

        # Need to fix simulation
        nonbonded_force.updateParametersInContext(self.simulation_solvent.context)

    def update_harmonic_bond(self, scale):
        """Scale solvent only system nonbonded interaction:
        - LJ epsilon by `scale`
        - Coulomb charges by `sqrt(scale)`
        - charge product is scaled by `scale`
        """

        harmonic_bond_force = self.harmonic_bond_force

        for i, index in enumerate(self.init_harmonic_bond_index):
            p1, p2, r_0, k = self.init_harmonic_bond_value[i]
            harmonic_bond_force.setBondParameters(
                index, p1, p2, [r_0, k * scale]
            )

        harmonic_bond_force.updateParametersInContext(self.simulation.context)


    def update_harmonic_angle(self, scale):
        """Scale solvent only system nonbonded interaction:
        - LJ epsilon by `scale`
        - Coulomb charges by `sqrt(scale)`
        - charge product is scaled by `scale`
        """

        harmonic_angle_force = self.harmonic_angle_force

        for i, index in enumerate(self.init_harmonic_angle_index):
            p1, p2, p3, theta_0, k = self.init_harmonic_angle_value[i]
            harmonic_angle_force.setAngleParameters(
                index, p1, p2, p3, [theta_0, k * scale]
            )

        harmonic_angle_force.updateParametersInContext(self.simulation.context)

    def scale_nonbonded_bonded(self, scale):
        """Scale solvent nonbonded potential and
        solute torsion potential
        """

        self.scale = scale
        self.update_nonbonded(scale)
        self.update_nonbonded_solvent(scale)
        self.update_torsions(scale)
        self.update_harmonic_bond(scale)
        self.update_harmonic_angle(scale)

    def compute_all_energies(self):
        """Extract solute potential energy and solute-solvent interactions."""


        # Extract non bonded forces
        solute_force, solvent_force = self.compute_solute_solvent_system_energy()

        E_solvent = 0 * unit.kilojoules_per_mole
        E_solute = 0 * unit.kilojoules_per_mole

        for i, force in solute_force.items():
            if force["name"] == "NonbondedForce":
                solute_nb = force["energy"]
                E_solute += force["energy"]


        for i, force in solvent_force.items():
            if force["name"] == "NonbondedForce":
                solvent_nb = force["energy"]
                E_solvent += force["energy"]

        # Extract bonded forces

        system_force = get_forces(self.system, self.simulation)

        solvent_bond_flag = False
        solvent_angle_flag = False
        solvent_torsion_flag = False 


        for i, force in system_force.items():
            if force["name"] == "NonbondedForce":
                all_nb = force["energy"]
            # flag to get first component of
            # forces (the solute one)
            elif force["name"] == "CustomBondForce":
                if not solvent_bond_flag:
                    E_solute += force["energy"]
                    solvent_bond_flag = True
                else:
                    E_solvent += force["energy"]
            elif force["name"] == "CustomAngleForce":
                if not solvent_angle_flag:
                    E_solute += force["energy"]
                    solvent_angle_flag = True
                else:
                    E_solvent += force["energy"]
            elif force["name"] == "CustomTorsionForce":
                if not solvent_torsion_flag:
                    E_solute += force["energy"]
                    solvent_torsion_flag = True
                else:
                    E_solvent += force["energy"]



        # Non scaled solvent-solute_non bonded:
        solvent_solute_nb = all_nb - solute_nb - solvent_nb
        # Scaled non bonded
        # solvent_solute_nb *= (1 / self.scale)**0.5
        # solute_nb *= 1 / self.scale

        return (
            E_solute,
            (1 / self.scale) * E_solvent,
            (1 / self.scale) ** 0.5 * solvent_solute_nb,
        )

    def get_customPotEnergie(self):
        """Extract solute potential energy and solute-solvent interactions."""

        E_solute_scaled, _, _, solvent_solute_nb = self.compute_all_energies()

        return E_solute_scaled + 0.5 * (1 / self.scale) ** 0.5 * solvent_solute_nb


def run_rest1(
    sys_rest1,
    generic_name,
    tot_steps,
    dt,
    save_step_dcd=100000,
    save_step_log=500,
    save_step_rest1=500,
    overwrite=False,
    remove_reporters=True,
    add_REST1_reporter=True,
    save_checkpoint_steps=None,
):
    """
    Run REST1 simulation

    Parameters
    ----------
    sys_rest1 : Rest1 object
        System to run
    generic_name : str
        Generic name for output files
    tot_steps : int
        Total number of steps to run
    dt : float
        Time step in fs
    save_step_dcd : int, optional
        Step to save dcd file, by default 100000
    save_step_log : int, optional
        Step to save log file, by default 500
    save_step_rest1 : int, optional
        Step to save rest1 file, by default 500
    overwrite : bool, optional
        If True, overwrite previous files, by default False
    save_checkpoint_steps : int, optional
        Step to save checkpoint file, by default None

    """

    if not overwrite and os.path.isfile(generic_name + "_final.xml"):
        logger.info(
            f"File {generic_name}_final.xml exists already, skip simulate() step"
        )
        sys_rest1.simulation.loadState(generic_name + "_final.xml")
        return

    new_reporter = []
    if add_REST1_reporter:
        new_reporter = [
            Rest1Reporter(f"{generic_name}_rest1.csv", save_step_rest1, sys_rest1)
        ]

    simulate(
        sys_rest1.simulation,
        sys_rest1.topology,
        tot_steps,
        dt,
        generic_name,
        additional_reporters=new_reporter,
        save_step_log=save_step_log,
        save_step_dcd=save_step_dcd,
        remove_reporters=remove_reporters,
        save_checkpoint_steps=save_checkpoint_steps,
    )


if __name__ == "__main__":
    # Check energy decompostion is correct:

    from openmm.app import ForceField, PME, HBonds
    from openmm import LangevinMiddleIntegrator
    import tools
    import pdb_numpy

    # Whole system:
    OUT_PATH = "/mnt/Data_3/SST2_clean/tmp"
    name = "2HPL"

    forcefield = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")

    dt = 2 * unit.femtosecond
    temperature = 300 * unit.kelvin
    friction = 1 / unit.picoseconds

    # SYSTEM

    equi_coor = pdb_numpy.Coor(f"src/SST2/tests/inputs/{name}_equi_water.pdb")
    solute = equi_coor.select_atoms("chain B")
    solute.write(f"{name}_only_pep.pdb", overwrite=True)
    solvant = equi_coor.select_atoms("not chain B")
    solvant.write(f"{name}_no_pep.pdb", overwrite=True)

    pdb = app.PDBFile(f"src/SST2/tests/inputs/{name}_equi_water.pdb")

    integrator = LangevinMiddleIntegrator(temperature, friction, dt)

    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=1 * unit.nanometers,
        constraints=HBonds,
    )

    simulation = setup_simulation(
        system, pdb.positions, pdb.topology, integrator, "CUDA"
    )

    print("Whole system energy")
    tools.print_forces(system, simulation)
    forces_sys = tools.get_forces(system, simulation)

    """
    nsteps=10000
    every_step = 500
    set_coor_pep = False

    scale = 1.0

    print('Timing %d steps of integration...' % nsteps)
    initial_time = time.time()
    for i in range(nsteps//every_step):
        #print('.', end='')
        simulation.step(every_step)
        scale *= 0.95
        
        #_update_custom_nonbonded(simulation, scale, solute_solvent_custom_nonbonded_force, init_nb_param)
        
        if set_coor_pep:
            tot_positions = simulation.context.getState(
                getPositions=True,
                getEnergy=True).getPositions()
            simulation_pep.context.setPositions(tot_positions[solute[0]:solute[-1]+1])
            forces = get_forces(system_pep, simulation_pep)
    print()
    final_time = time.time()
    elapsed_time = (final_time - initial_time) * unit.seconds
    integrated_time = nsteps * dt
    print(f'{nsteps} steps of {dt/unit.femtoseconds:.1f} fs timestep' +
          f'({integrated_time/unit.picoseconds:.3f} ps) took {elapsed_time/unit.seconds:.3f}'+
          f' s : {(integrated_time/elapsed_time)/(unit.nanoseconds/unit.day):.3f} ns/day')
    """

    # PEPTIDE system:

    pdb_pep = app.PDBFile(f"{name}_only_pep.pdb")

    integrator_pep = LangevinMiddleIntegrator(temperature, friction, dt)

    system_pep = forcefield.createSystem(
        pdb_pep.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=1 * unit.nanometers,
        constraints=HBonds,
    )

    simulation_pep = setup_simulation(
        system_pep, pdb_pep.positions, pdb_pep.topology, integrator_pep, "CUDA"
    )

    print("Peptide forces:")
    tools.print_forces(system_pep, simulation_pep)
    forces_pep = tools.get_forces(system_pep, simulation_pep)

    # NO Peptide system

    pdb_no_pep = app.PDBFile(f"{name}_no_pep.pdb")

    integrator_no_pep = LangevinMiddleIntegrator(temperature, friction, dt)

    system_no_pep = forcefield.createSystem(
        pdb_no_pep.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=1 * unit.nanometers,
        constraints=HBonds,
    )

    simulation_no_pep = setup_simulation(
        system_no_pep,
        pdb_no_pep.positions,
        pdb_no_pep.topology,
        integrator_no_pep,
        "CUDA",
    )

    print("No Peptide forces:")
    tools.print_forces(system_no_pep, simulation_no_pep)
    forces_no_pep = tools.get_forces(system_no_pep, simulation_no_pep)

    ####################
    # ## REST1 test ####
    ####################

    # Get indices of the three sets of atoms.
    all_indices = [int(i.index) for i in pdb.topology.atoms()]
    solvent_indices = [
        int(i.index) for i in pdb.topology.atoms() if not (i.residue.chain.id in ["B"])
    ]
    solute_indices = [
        int(i.index) for i in pdb.topology.atoms() if i.residue.chain.id in ["B"]
    ]

    print(f" {len(solute_indices)} atom in solute group")

    integrator_rest = LangevinMiddleIntegrator(temperature, friction, dt)

    test = REST1(system, pdb, forcefield, solute_indices, integrator_rest)

    ####################
    # ## REST1 test ####
    ####################

    print("REST1 forces 300K:")
    tools.print_forces(test.system, test.simulation)
    forces_rest1 = tools.get_forces(test.system, test.simulation)

    print("\nCompare energy rest1 vs. classic:\n")
    E_rest1_bond = forces_rest1[2]['energy'] + forces_rest1[3]['energy']
    print(
        f"HarmonicBondForce    {E_rest1_bond/forces_sys[0]['energy']:.5e}"
    )
    E_rest1_angle = forces_rest1[4]['energy'] + forces_rest1[5]['energy']
    print(
        f"HarmonicAngleForce   {E_rest1_angle/forces_sys[4]['energy']:.5e}"
    )

    torsion_force = (
        forces_rest1[6]["energy"]
        + forces_rest1[7]["energy"]
    )

    print(f"PeriodicTorsionForce {torsion_force/forces_sys[2]['energy']:.5e}")
    print(
        f"NonbondedForce       {forces_rest1[0]['energy']/forces_sys[1]['energy']:.5e}"
    )
    print(
        f"Total                {forces_rest1[10]['energy']/forces_sys[6]['energy']:.5e}"
    )

    (
        E_solute,
        E_solvent,
        solvent_solute_nb,
    ) = test.compute_all_energies()

    print(f"E_solute             {E_solute}")
    print(f"E_solvent            {E_solvent}")
    print(f"solvent_solute_nb    {solvent_solute_nb}")

    total_ener = E_solute + E_solvent + solvent_solute_nb

    print(
        f"Total Computed       {total_ener/forces_sys[6]['energy']:.5e}"
    )


    print("\nCompare Solute:\n")
    torsion_force = forces_rest1[4]["energy"] + forces_rest1[5]["energy"]
    print(f"Total {E_solute/forces_pep[6]['energy']:.5e}")

    print("\nCompare Solvent:\n")
    torsion_force = forces_rest1[6]["energy"]
    print(f"Total {E_solvent/forces_no_pep[6]['energy']:.5e}")

    print("\nCompare nonbond energy rest1 vs. no pep+pep+solvent_solute_nb:\n")
    non_bonded = (
        solvent_solute_nb + forces_pep[1]["energy"] + forces_no_pep[1]["energy"]
    )
    print(f"NonbondedForce       {non_bonded/forces_sys[1]['energy']:.5e}")


    scale = 0.5
    test.scale_nonbonded_bonded(scale)
    print("REST1 forces 600K:")
    tools.print_forces(test.system, test.simulation)
    forces_rest1 = tools.get_forces(test.system, test.simulation)
    (
        E_solute_new,
        E_solvent_new,
        solvent_solute_nb_new,
    ) = test.compute_all_energies()
    print(f"ratio new/old E_solute             {E_solute/E_solute_new}")
    print(f"ratio new/old E_solvent            {E_solvent/E_solvent_new}")
    print(f"ratio new/old solvent_solute_nb    {solvent_solute_nb/solvent_solute_nb_new}")

    print("\nCompare not scaled energy rest1 vs. classic:\n")
    print(
        f"HarmonicBondForce    {forces_rest1[0]['energy']/forces_sys[0]['energy']:.5e}"
    )
    print(
        f"HarmonicAngleForce   {forces_rest1[3]['energy']/forces_sys[4]['energy']:.5e}"
    )
    print("Compare scaled energy:")
    torsion_force = (
        forces_rest1[4]["energy"] * scale
        + forces_rest1[5]["energy"]
        + forces_rest1[6]["energy"]
    )
    print(f"PeriodicTorsionForce {torsion_force/forces_sys[2]['energy']:.5e}")
    print(
        f"NonbondedForce       {forces_rest1[1]['energy']/forces_sys[1]['energy']:.5e}"
    )
    print(
        f"Total                {forces_rest1[10]['energy']/forces_sys[6]['energy']:.5e}"
    )

    print("\nCompare torsion energy rest1 vs. pep:\n")
    torsion_force = forces_rest1[4]["energy"] + forces_rest1[5]["energy"]
    print(f"PeriodicTorsionForce {torsion_force/forces_pep[2]['energy']:.5e}")

    print("\nCompare torsion energy rest1 vs. no pep:\n")
    torsion_force = forces_rest1[6]["energy"]
    print(f"PeriodicTorsionForce {torsion_force/forces_no_pep[2]['energy']:.5e}")

    print("\nCompare nonbond energy rest1 vs. no pep+pep+solvent_solute_nb:\n")
    non_bonded = (
        solvent_solute_nb + forces_pep[2]["energy"] + forces_no_pep[2]["energy"]
    )
    print(f"NonbondedForce       {non_bonded/forces_sys[2]['energy']:.5e}")

    solute_scaled_force = forces_rest1[4]["energy"] + forces_pep[2]["energy"]
    print(f"E_solute_scaled      {solute_scaled_force/E_solute_scaled:.5e}")
    solute_not_scaled_force = (
        forces_rest1[5]["energy"] + forces_pep[0]["energy"] + +forces_pep[1]["energy"]
    )
    print(f"E_solute_not_scaled  {non_bonded/forces_sys[2]['energy']:.5e}")

    print(f"E_solvent            {E_solvent/forces_no_pep[6]['energy']:.5e}")


    """
    vmd test_2HPL/2HPL_em_water.pdb test_2HPL/2HPL_equi_water.dcd -m 2HPL.pdb
    pbc wrap -molid 0 -first 0 -last last -compound fragment -center com -centersel "chain A and protein" -orthorhombic
    """
