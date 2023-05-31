import copy
import math
from io import StringIO
import sys

from openmm.app import PDBFile, ForceField, HBonds, Simulation, PME

from openmm import (
    LangevinMiddleIntegrator,
    MonteCarloBarostat,
    Platform,
    unit,
    openmm,
)

import pdb_numpy.format

from .setup import setup_simulation, create_system_simulation

"""
To fix:

    - location of the pdb file save (solute.pdb and solvent.pdb)
    - Add some test to ensure that indexes of solute and solvant are correct

"""


class Rest2Reporter(object):
    def __init__(self, file, reportInterval, rest2):
        self._out = open(file, "w", buffering=1)
        self._out.write(
            "ps,Solute scaled(kJ/mol),Solute not scaled(kJ/mol),Solvent(kJ/mol),Solute-Solvent(kJ/mol)\n"
        )
        self._reportInterval = reportInterval
        self._rest2 = rest2

    def __del__(self):
        self._out.close()

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, False, False, False, False, None)

    def report(self, simulation, state):

        energies = self._rest2.compute_all_energies()

        # E_solute_scaled, E_solute_not_scaled, E_solvent, solvent_solute_nb

        time = state.getTime().value_in_unit(unit.picosecond)
        self._out.write(
            f"{time},{energies[0]._value},{energies[1]._value},{energies[2]._value},{energies[3]._value}\n"
        )


class REST2:
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
        precision="single",
        dt=2*unit.femtosecond,
        friction=1/unit.picoseconds,
        nonbondedMethod=PME,
        nonbondedCutoff=1 * unit.nanometers,
        constraints=HBonds,
        rigidWater=True,
        ewaldErrorTolerance=0.0005,
        hydrogenMass=3.0 * unit.amu,
    ):

        self.system = system
        self.positions = pdb.positions
        self.topology = pdb.topology
        self.solute_index = solute_index
        self.solvent_index = list(
            set(range(self.system.getNumParticles())).difference(set(self.solute_index))
        )
        self.system_forces = {
            type(force).__name__: force for force in self.system.getForces()
        }
        self.scale = 1.0

        # Extract solute nonbonded index and values
        self.find_solute_nb_index()
        # Separate solute torsion from the solvent
        self.separate_torsion_pot()
        # Extract solute torsions index and values
        self.find_torsions()
        # Create separate solute and solvent simulation
        if self.solvent_index and self.solute_index:
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
            # Extract solute nonbonded index and values from the solute_only system
            self.find_nb_solute_system()
        self.setup_simulation(
            integrator,
            temperature=temperature,
            pressure=pressure,
            barostatInterval=barostatInterval,
            platform_name=platform_name,
            precision=precision,
        )

    def create_solute_solvent_simulation(
        self,
        forcefield,
        solute_out_pdb="solute.pdb",
        solvent_out_pdb="solvent.pdb",
        nonbondedMethod=PME,
        nonbondedCutoff=1 * unit.nanometers,
        constraints=HBonds,
        platform_name="CUDA",
        rigidWater=True,
        ewaldErrorTolerance=0.0005,
        hydrogenMass=3.0 * unit.amu,
        friction=1/unit.picoseconds,
        dt=2*unit.femtosecond,
    ):
        """Extract solute only and solvent only coordinates.
        A sytem and a simulation is then created for both systems.
        """

        # Save pdb coordinates to read them with pdb_numpy

        # Redirect stdout in the variable new_stdout:
        old_stdout = sys.stdout
        stdout = new_stdout = StringIO()
        # In case of dummy atoms (position restraints, ...)
        # It has to be removed from pdb files
        top_num_atom = self.topology.getNumAtoms()

        PDBFile.writeFile(self.topology, self.positions[:top_num_atom], stdout, True)
        sys.stdout = old_stdout

        # Read
        solute_solvent_coor = pdb_numpy.format.pdb.parse(
            new_stdout.getvalue().split("\n")
        )

        # Separate coordinates in two pdb files:
        solute_coor = solute_solvent_coor.select_index(self.solute_index)
        solute_coor.write(solute_out_pdb, check_file_out=False)

        solvent_coor = solute_solvent_coor.select_index(self.solvent_index)
        solvent_coor.write(solvent_out_pdb, check_file_out=False)

        # Create system and simulations:
        self.system_solute, self.simulation_solute = create_system_simulation(
            solute_out_pdb,
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

    def setup_simulation(
        self,
        integrator,
        temperature=300 * unit.kelvin,
        pressure=1.0 * unit.atmospheres,
        barostatInterval=25,
        platform_name="CUDA",
        precision="single",
    ):
        """Add the simulation object."""

        # Add PT MonteCarlo barostat
        self.system.addForce(
            MonteCarloBarostat(pressure, temperature, barostatInterval)
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

    def separate_bond_pot(self):
        """Useless in the REST2 case as solute potential energy
        is obtain from the solute only system.
        """

        energy_expression = "(k/2)*(r-length)^2;"

        # Create the Solvent bond
        solvent_bond_force = openmm.CustomBondForce(energy_expression)
        solvent_bond_force.addPerBondParameter("length")
        solvent_bond_force.addPerBondParameter("k")

        # Create the Solute bond
        solute_bond_force = openmm.CustomBondForce(energy_expression)
        solute_bond_force.addPerBondParameter("length")
        solute_bond_force.addPerBondParameter("k")

        original_bond_force = self.system_forces["HarmonicBondForce"]

        for i in range(original_bond_force.getNumBonds()):
            p1, p2, length, k = original_bond_force.getBondParameters(i)
            # print(p1, p2)

            if p1 in self.solute_index and p2 in self.solute_index:
                solute_bond_force.addBond(p1, p2, [length, k])
            elif p1 not in self.solute_index and p2 not in self.solute_index:
                solvent_bond_force.addBond(p1, p2, [length, k])
            else:
                print("Wrong bond !")
                exit()

        self.system.addForce(solute_bond_force)
        self.system.addForce(solvent_bond_force)

    def separate_angle_pot(self):
        """Useless in the REST2 case as solute potential energy
        is obtain from the solute only system.
        """

        energy_expression = "(k/2)*(theta-theta0)^2;"

        # Create the Solvent bond
        solvent_angle_force = openmm.CustomAngleForce(energy_expression)
        solvent_angle_force.addPerAngleParameter("theta0")
        solvent_angle_force.addPerAngleParameter("k")

        # Create the Solute bond
        solute_angle_force = openmm.CustomAngleForce(energy_expression)
        solute_angle_force.addPerAngleParameter("theta0")
        solute_angle_force.addPerAngleParameter("k")

        original_angle_force = self.system_forces["HarmonicAngleForce"]

        for i in range(original_angle_force.getNumAngles()):
            p1, p2, p3, theta0, k = original_angle_force.getAngleParameters(i)
            if (
                p1 in self.solute_index
                and p2 in self.solute_index
                and p3 in self.solute_index
            ):
                solute_angle_force.addAngle(p1, p2, p3, [theta0, k])
            elif (
                p1 not in self.solute_index
                and p2 not in self.solute_index
                and p3 not in self.solute_index
            ):
                solvent_angle_force.addAngle(p1, p2, p3, [theta0, k])
            else:
                print("Wrong Angle !")
                exit()

        self.system.addForce(solute_angle_force)
        self.system.addForce(solvent_angle_force)

    def separate_torsion_pot(self):
        """Use in the REST2 case as it avoid to modify
        twice the torsion terms in the rest2 system and
        in the solute system.

        Torsion potential is separate in two groups, one for
        the solute (scaled) and one for the solvent and not scaled solute torsion.

        As improper angles are not supposed to be scaled, here we extract only
        the proper torsion angles.

        To identify proper angles we use a trick from:
        https://github.com/maccallumlab/meld/blob/master/meld/runner/transform/rest2.py

        The original torsion potential is deleted.
        """

        energy_expression = "k*(1+cos(period*theta-phase));"

        # Create the Solvent bond and not scaled solute torsion
        solvent_torsion_force = openmm.CustomTorsionForce(energy_expression)
        solvent_torsion_force.addPerTorsionParameter("period")
        solvent_torsion_force.addPerTorsionParameter("phase")
        solvent_torsion_force.addPerTorsionParameter("k")

        # Create the Solute bond
        solute_scaled_torsion_force = openmm.CustomTorsionForce(energy_expression)
        solute_scaled_torsion_force.addPerTorsionParameter("period")
        solute_scaled_torsion_force.addPerTorsionParameter("phase")
        solute_scaled_torsion_force.addPerTorsionParameter("k")

        solute_not_scaled_torsion_force = openmm.CustomTorsionForce(energy_expression)
        solute_not_scaled_torsion_force.addPerTorsionParameter("period")
        solute_not_scaled_torsion_force.addPerTorsionParameter("phase")
        solute_not_scaled_torsion_force.addPerTorsionParameter("k")

        original_torsion_force = self.system_forces["PeriodicTorsionForce"]

        bond_idxs = [sorted([i.index, j.index]) for i, j in self.topology.bonds()]

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

            not_improper = (
                sorted([p1, p2]) in bond_idxs
                and sorted([p2, p3]) in bond_idxs
                and sorted([p3, p4]) in bond_idxs
            )

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

            if solute_in and not_improper:
                solute_scaled_torsion_force.addTorsion(
                    p1, p2, p3, p4, [periodicity, phase, k]
                )
            elif solute_in:
                solute_not_scaled_torsion_force.addTorsion(
                    p1, p2, p3, p4, [periodicity, phase, k]
                )
            elif solvent_in:
                solvent_torsion_force.addTorsion(
                    p1, p2, p3, p4, [periodicity, phase, k]
                )
            else:
                print("Wrong Torsion !")
                exit()

        self.solute_torsion_force = solute_scaled_torsion_force

        self.system.addForce(solute_scaled_torsion_force)
        self.system.addForce(solute_not_scaled_torsion_force)
        self.system.addForce(solvent_torsion_force)

        print("- Delete original Torsion Forces")

        for count, force in enumerate(self.system.getForces()):
            if isinstance(force, openmm.PeriodicTorsionForce):
                self.system.removeForce(count)

    def find_torsions(self):
        """Extract the initial solute torsion index and values.
        As improper angles are not supposed to be scaled, here we extract only
        the proper torsion angles.

        To identify proper angles we use a trick from:
        https://github.com/maccallumlab/meld/blob/master/meld/runner/transform/rest2.py

        """

        self.init_torsions_index = []
        self.init_torsions_value = []

        torsion_force = self.solute_torsion_force

        bond_idxs = [sorted([i.index, j.index]) for i, j in self.topology.bonds()]

        for i in range(torsion_force.getNumTorsions()):
            (
                p1,
                p2,
                p3,
                p4,
                [periodicity, phase, k],
            ) = torsion_force.getTorsionParameters(i)

            # Probably useless, to check
            not_improper = (
                sorted([p1, p2]) in bond_idxs
                and sorted([p2, p3]) in bond_idxs
                and sorted([p3, p4]) in bond_idxs
            )

            # Probably useless, to check
            not_solvent = (
                p1 in self.solute_index
                and p2 in self.solute_index
                and p3 in self.solute_index
                and p4 in self.solute_index
            )

            if not_improper and not_solvent:
                self.init_torsions_index.append(i)
                self.init_torsions_value.append([p1, p2, p3, p4, periodicity, phase, k])

        print(f"Solute torsion number : {len(self.init_torsions_index)}")

    def update_torsions(self, scale):
        """Scale system solute torsion by a scale factor."""

        torsion_force = self.solute_torsion_force

        for i, index in enumerate(self.init_torsions_index):
            p1, p2, p3, p4, periodicity, phase, k = self.init_torsions_value[i]
            torsion_force.setTorsionParameters(
                index, p1, p2, p3, p4, [periodicity, phase, k * scale]
            )

        torsion_force.updateParametersInContext(self.simulation.context)

    def find_solute_nb_index(self):
        """Extract initial solute nonbonded indexes and values (charge, sigma, epsilon).
        Extract also exeption indexes and values (chargeprod, sigma, epsilon)
        """

        nonbonded_force = self.system_forces["NonbondedForce"]

        # Copy particles
        self.init_nb_param = []
        for particle_index in range(nonbonded_force.getNumParticles()):
            [charge, sigma, epsilon] = nonbonded_force.getParticleParameters(
                particle_index
            )
            self.init_nb_param.append([charge, sigma, epsilon])

        # Copy solute-solute exclusions
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
            if iatom in self.solute_index and jatom in self.solute_index:
                self.init_nb_exept_index.append(exception_index)
                self.init_nb_exept_value.append(
                    [iatom, jatom, chargeprod, sigma, epsilon]
                )

    def find_nb_solute_system(self):
        """Extract in the solute only system:
        - exeption indexes and values (chargeprod, sigma, epsilon)

        solute nonbonded values are not extracted as they are identical to
        the main system. Indexes are [0 :len(nonbonded values)]

        Exeption values are stored as indexes [iatom, jatom] are different.
        """

        nonbonded_force = self.system_forces_solute["NonbondedForce"]

        # Copy particles
        self.init_nb_exept_solute_value = []
        for exception_index in range(nonbonded_force.getNumExceptions()):
            [
                iatom,
                jatom,
                chargeprod,
                sigma,
                epsilon,
            ] = nonbonded_force.getExceptionParameters(exception_index)
            self.init_nb_exept_solute_value.append(
                [iatom, jatom, chargeprod, sigma, epsilon]
            )

    def update_nonbonded(self, scale):
        """Scale system nonbonded interaction:
        - LJ epsilon by `scale`
        - Coulomb charges by `sqrt(scale)`
        - charge product is scaled by `scale`
        """

        nonbonded_force = self.system_forces["NonbondedForce"]

        for i in self.solute_index:
            q, sigma, eps = self.init_nb_param[i]
            nonbonded_force.setParticleParameters(
                i, q * math.sqrt(scale), sigma, eps * scale
            )

        for i in range(len(self.init_nb_exept_index)):
            index = self.init_nb_exept_index[i]
            p1, p2, q, sigma, eps = self.init_nb_exept_value[i]
            nonbonded_force.setExceptionParameters(
                index, p1, p2, q * scale, sigma, eps * scale
            )
        # Need to fix simulation
        nonbonded_force.updateParametersInContext(self.simulation.context)

    def update_nonbonded_solute(self, scale):
        """Scale solute only system nonbonded interaction:
        - LJ epsilon by `scale`
        - Coulomb charges by `sqrt(scale)`
        - charge product is scaled by `scale`
        """

        nonbonded_force = self.system_forces_solute["NonbondedForce"]

        for i in range(len(self.solute_index)):
            q, sigma, eps = self.init_nb_param[i]
            nonbonded_force.setParticleParameters(
                i, q * math.sqrt(scale), sigma, eps * scale
            )

        for i in range(len(self.init_nb_exept_index)):
            p1, p2, q, sigma, eps = self.init_nb_exept_solute_value[i]
            nonbonded_force.setExceptionParameters(
                i, p1, p2, q * scale, sigma, eps * scale
            )
        # Need to fix simulation
        nonbonded_force.updateParametersInContext(self.simulation_solute.context)

    def scale_nonbonded_torsion(self, scale):
        """Scale solute nonbonded potential and
        solute torsion potential
        """

        self.scale = scale
        self.update_nonbonded(scale)
        self.update_nonbonded_solute(scale)
        self.update_torsions(scale)

    def compute_all_energies(self):
        """Extract solute potential energy and solute-solvent interactions."""

        solute_force, solvent_force = self.compute_solute_solvent_system_energy()

        # print("Solute:", solute_force)
        # print("Solvent:", solvent_force)

        E_solute_not_scaled = 0 * unit.kilojoules_per_mole
        E_solute_scaled = 0 * unit.kilojoules_per_mole
        solute_not_scaled_term = ["HarmonicBondForce", "HarmonicAngleForce"]

        for i, force in solute_force.items():
            if force["name"] == "NonbondedForce":
                solute_nb = force["energy"]
                E_solute_scaled += force["energy"]
            elif force["name"] in solute_not_scaled_term:
                E_solute_not_scaled += force["energy"]

        solvent_term = [
            "HarmonicBondForce",
            "HarmonicAngleForce",
            "NonbondedForce",
            "PeriodicTorsionForce",
        ]
        E_solvent = 0 * unit.kilojoules_per_mole
        for i, force in solvent_force.items():
            if force["name"] == "NonbondedForce":
                solvent_nb = force["energy"]
            if force["name"] in solvent_term:
                E_solvent += force["energy"]

        system_force = get_forces(self.system, self.simulation)

        # print("System:", system_force)

        E_tot = 0 * unit.kilojoules_per_mole
        solute_torsion_scaled_flag = True
        solute_torsion_not_scaled_flag = False
        system_term = [
            "HarmonicBondForce",
            "HarmonicAngleForce",
            "PeriodicTorsionForce",
            "CustomTorsionForce",
        ]

        for i, force in system_force.items():
            if force["name"] == "NonbondedForce":
                all_nb = force["energy"]
            # Torsion flag to get first component of dihedral
            # forces (the solute one)
            if force["name"] == "CustomTorsionForce" and solute_torsion_scaled_flag:
                E_tot += force["energy"]
                E_solute_scaled += force["energy"]
                solute_torsion_scaled_flag = False
                solute_torsion_not_scaled_flag = True
            if force["name"] == "CustomTorsionForce" and solute_torsion_not_scaled_flag:
                E_tot += force["energy"]
                E_solute_not_scaled += force["energy"]
                solute_torsion_not_scaled_flag = False
            elif force["name"] in system_term:
                E_tot += force["energy"]

        # print(f"1 Nonbonde solute : {solute_nb} solvent : {solvent_nb} system: {all_nb}")

        # Non scaled solvent-solute_non bonded:
        solvent_solute_nb = all_nb - solute_nb - solvent_nb
        # Scaled non bonded
        # solvent_solute_nb *= (1 / self.scale)**0.5
        # solute_nb *= 1 / self.scale

        E_tot += solvent_solute_nb + solute_nb + solvent_nb
        all_nb = solvent_solute_nb + solute_nb + solvent_nb

        # print(f"2 Nonbonde solute : {solute_nb} solvent : {solvent_nb} system: {all_nb}")

        # print(f'Energie (kJ/mol) Solute = {E_solute._value:8.3f} '\
        #       f'Solvent = {E_solvent._value:8.3f} '\
        #       f'Total = {E_tot._value:8.3f} '\
        #       f'Solute-Solvent = {(E_tot - E_solvent - E_solute)._value:8.3f} ')

        # return(E_solute_scaled, E_solute_not_scaled, E_solvent, solvent_solute_nb)
        return (
            (1 / self.scale) * E_solute_scaled,
            E_solute_not_scaled,
            E_solvent,
            (1 / self.scale) ** 0.5 * solvent_solute_nb,
        )

    def get_customPotEnergie(self):
        """Extract solute potential energy and solute-solvent interactions."""

        E_solute_scaled, _, _, solvent_solute_nb = self.compute_all_energies()

        return E_solute_scaled + 0.5 * (1 / self.scale) ** 0.5 * solvent_solute_nb

    ###########################################################
    ###  OLD FUNCTION To ensure nb calculation are correct  ###
    ###########################################################
    def add_custom_LJ_forces(self):
        """
        Create CustomNonbondedForce to capture solute-solute and solute-solvent interactions.
        Assumes PME is in use.

        Taken from:
        https://github.com/openmm/openmm/pull/2014
        """
        # nonbonded_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)

        nonbonded_force = self.system_forces["NonbondedForce"]

        # Determine PME parameters from nonbonded_force
        cutoff_distance = nonbonded_force.getCutoffDistance()
        [alpha_ewald, nx, ny, nz] = nonbonded_force.getPMEParameters()
        if (alpha_ewald / alpha_ewald.unit) == 0.0:
            # If alpha is 0.0, alpha_ewald is computed by OpenMM from from the error tolerance
            tol = nonbonded_force.getEwaldErrorTolerance()
            alpha_ewald = (1.0 / cutoff_distance) * math.sqrt(-math.log(2.0 * tol))
        print(alpha_ewald)

        # Create CustomNonbondedForce
        ONE_4PI_EPS0 = 138.935456
        energy_expression = "4*epsilon*((sigma/r)^12 - (sigma/r)^6);"
        # energy_expression += "epsilon = epsilon1*epsilon2;" # Why not epsilon = sqrt(epsilon1*epsilon2)
        energy_expression += "epsilon = sqrt(epsilon1*epsilon2);"  # Why not epsilon = sqrt(epsilon1*epsilon2)
        energy_expression += "sigma = 0.5*(sigma1+sigma2);"
        custom_nonbonded_force = openmm.CustomNonbondedForce(energy_expression)
        custom_nonbonded_force.addPerParticleParameter("sigma")
        custom_nonbonded_force.addPerParticleParameter("epsilon")
        # custom_nonbonded_force.addPerParticleParameter('soluteFlag')
        # custom_nonbonded_force.addInteractionGroup(solute, solvent)

        # Configure force
        custom_nonbonded_force.setNonbondedMethod(
            openmm.CustomNonbondedForce.CutoffPeriodic
        )
        custom_nonbonded_force.setCutoffDistance(cutoff_distance)
        custom_nonbonded_force.setUseLongRangeCorrection(True)  # True

        switch_flag = nonbonded_force.getUseSwitchingFunction()
        print("switch", switch_flag)
        if switch_flag:
            custom_nonbonded_force.setUseSwitchingFunction(True)
            switching_distance = nonbonded_force.getSwitchingDistance()
            custom_nonbonded_force.setSwitchingDistance(switching_distance)
        else:  # Truncated
            custom_nonbonded_force.setUseSwitchingFunction(False)

        # Create CustomBondForce for exceptions
        energy_expression = "4*epsilon*((sigma/r)^12 - (sigma/r)^6)"
        custom_bond_force = openmm.CustomBondForce(energy_expression)
        custom_bond_force.addPerBondParameter("sigma")
        custom_bond_force.addPerBondParameter("epsilon")

        # Copy particles
        for particle_index in range(nonbonded_force.getNumParticles()):
            [charge, sigma, epsilon] = nonbonded_force.getParticleParameters(
                particle_index
            )
            if sigma == 0:
                print(charge, sigma, epsilon)
            # solute_type = 1 if index in solute else 0
            # solute_type = 1
            # custom_nonbonded_force.addParticle([charge, sigma, epsilon, solute_type])
            custom_nonbonded_force.addParticle([sigma, epsilon])

        # Copy solute-solute exclusions
        for exception_index in range(nonbonded_force.getNumExceptions()):
            [
                iatom,
                jatom,
                chargeprod,
                sigma,
                epsilon,
            ] = nonbonded_force.getExceptionParameters(exception_index)
            custom_nonbonded_force.addExclusion(iatom, jatom)
            # if (iatom in solute) and (jatom in solute):
            custom_bond_force.addBond(iatom, jatom, [sigma, epsilon])

        self.system.addForce(custom_nonbonded_force)
        self.system.addForce(custom_bond_force)

    def add_custom_Coulomb_forces(self):
        """
        Create CustomNonbondedForce to capture solute-solute and solute-solvent interactions.
        Assumes PME is in use.

        Taken from:
        https://github.com/openmm/openmm/pull/2014
        """
        # nonbonded_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)

        nonbonded_force = self.system_forces["NonbondedForce"]

        # Determine PME parameters from nonbonded_force
        cutoff_distance = nonbonded_force.getCutoffDistance()
        [alpha_ewald, nx, ny, nz] = nonbonded_force.getPMEParameters()
        if (alpha_ewald / alpha_ewald.unit) == 0.0:
            # If alpha is 0.0, alpha_ewald is computed by OpenMM from from the error tolerance
            tol = nonbonded_force.getEwaldErrorTolerance()
            alpha_ewald = (1.0 / cutoff_distance) * math.sqrt(-math.log(2.0 * tol))
        print(alpha_ewald)

        # Create CustomNonbondedForce
        ONE_4PI_EPS0 = 138.935456
        energy_expression = "ONE_4PI_EPS0*chargeprod*erfc(alpha_ewald*r)/r;"
        energy_expression += "ONE_4PI_EPS0 = {:f};".format(
            ONE_4PI_EPS0
        )  # already in OpenMM units
        energy_expression += "chargeprod = charge1*charge2;"
        energy_expression += "alpha_ewald = {:f};".format(
            alpha_ewald.value_in_unit_system(unit.md_unit_system)
        )
        custom_nonbonded_force = openmm.CustomNonbondedForce(energy_expression)
        custom_nonbonded_force.addPerParticleParameter("charge")
        # custom_nonbonded_force.addPerParticleParameter('soluteFlag')
        # custom_nonbonded_force.addInteractionGroup(solute, solvent)

        # Configure force
        custom_nonbonded_force.setNonbondedMethod(
            openmm.CustomNonbondedForce.CutoffPeriodic
        )
        custom_nonbonded_force.setCutoffDistance(cutoff_distance)
        custom_nonbonded_force.setUseLongRangeCorrection(False)
        switch_flag = nonbonded_force.getUseSwitchingFunction()
        print("switch", switch_flag)
        if switch_flag:
            custom_nonbonded_force.setUseSwitchingFunction(True)
            switching_distance = nonbonded_force.getSwitchingDistance()
            custom_nonbonded_force.setSwitchingDistance(switching_distance)
        else:  # Truncated
            custom_nonbonded_force.setUseSwitchingFunction(False)

        # Create CustomBondForce for exceptions
        energy_expression = "ONE_4PI_EPS0*chargeProd_exceptions/r"
        energy_expression += (
            "- ONE_4PI_EPS0*chargeProd_product * erf(alpha_ewald * r) / r;"
        )
        energy_expression += "ONE_4PI_EPS0 = {:f};".format(
            ONE_4PI_EPS0
        )  # already in OpenMM units
        energy_expression += "alpha_ewald = {:f};".format(
            alpha_ewald.value_in_unit_system(unit.md_unit_system)
        )
        custom_bond_force = openmm.CustomBondForce(energy_expression)
        custom_bond_force.setUsesPeriodicBoundaryConditions(True)
        custom_bond_force.addPerBondParameter("chargeProd_exceptions")
        custom_bond_force.addPerBondParameter("chargeProd_product")

        # Copy particles
        for particle_index in range(nonbonded_force.getNumParticles()):
            [charge, sigma, epsilon] = nonbonded_force.getParticleParameters(
                particle_index
            )
            if sigma == 0:
                print(charge, sigma, epsilon)
            # solute_type = 1 if index in solute else 0
            # solute_type = 1
            # custom_nonbonded_force.addParticle([charge, sigma, epsilon, solute_type])
            custom_nonbonded_force.addParticle([charge])

        # Copy solute-solute exclusions
        for exception_index in range(nonbonded_force.getNumExceptions()):
            [
                iatom,
                jatom,
                chargeprod,
                sigma,
                epsilon,
            ] = nonbonded_force.getExceptionParameters(exception_index)
            custom_nonbonded_force.addExclusion(iatom, jatom)
            # if (iatom in solute) and (jatom in solute):

            # Compute chargeProd_product from original particle parameters
            p1_params = custom_nonbonded_force.getParticleParameters(iatom)
            p2_params = custom_nonbonded_force.getParticleParameters(jatom)
            # print(p1_params)
            chargeProd_product = p1_params[0] * p2_params[0]
            custom_bond_force.addBond(iatom, jatom, [chargeprod, chargeProd_product])

        self.system.addForce(custom_nonbonded_force)
        self.system.addForce(custom_bond_force)

    def add_reciprocal_force(self):
        standard_nonbonded_force = openmm.NonbondedForce()

        # Set nonbonded method and related attributes
        nonbonded_force = self.system_forces["NonbondedForce"]

        standard_nonbonded_method = nonbonded_force.getNonbondedMethod()
        standard_nonbonded_force.setNonbondedMethod(standard_nonbonded_method)
        if standard_nonbonded_method in [
            openmm.NonbondedForce.CutoffPeriodic,
            openmm.NonbondedForce.CutoffNonPeriodic,
        ]:
            epsilon_solvent = nonbonded_force.getReactionFieldDielectric()
            r_cutoff = nonbonded_force.getCutoffDistance()
            standard_nonbonded_force.setReactionFieldDielectric(epsilon_solvent)
            standard_nonbonded_force.setCutoffDistance(r_cutoff)
        elif standard_nonbonded_method in [
            openmm.NonbondedForce.PME,
            openmm.NonbondedForce.Ewald,
        ]:
            print("PME")
            [alpha_ewald, nx, ny, nz] = nonbonded_force.getPMEParameters()
            delta = nonbonded_force.getEwaldErrorTolerance()
            r_cutoff = nonbonded_force.getCutoffDistance()
            standard_nonbonded_force.setPMEParameters(alpha_ewald, nx, ny, nz)
            standard_nonbonded_force.setEwaldErrorTolerance(delta)
            standard_nonbonded_force.setCutoffDistance(r_cutoff)
        # Set the use of dispersion correction
        if nonbonded_force.getUseDispersionCorrection():
            print("Dispersion Correction")
            standard_nonbonded_force.setUseDispersionCorrection(True)
        else:
            standard_nonbonded_force.setUseDispersionCorrection(False)

        # Set the use of switching function
        if nonbonded_force.getUseSwitchingFunction():
            print("Switch")
            switching_distance = nonbonded_force.getSwitchingDistance()
            standard_nonbonded_force.setUseSwitchingFunction(True)
            standard_nonbonded_force.setSwitchingDistance(switching_distance)
        else:
            standard_nonbonded_force.setUseSwitchingFunction(False)

        # Disable direct space interactions
        standard_nonbonded_force.setIncludeDirectSpace(False)

        # Iterate over particles in original nonbonded force and copy to the new nonbonded force
        for particle_idx in range(nonbonded_force.getNumParticles()):
            # Get particle terms
            q, sigma, epsilon = nonbonded_force.getParticleParameters(particle_idx)

            # Add particle
            standard_nonbonded_force.addParticle(q, sigma, epsilon)

        for exception_idx in range(nonbonded_force.getNumExceptions()):
            # Get particle indices and exception terms
            p1, p2, chargeProd, sigma, epsilon = nonbonded_force.getExceptionParameters(
                exception_idx
            )

            # Add exception
            standard_nonbonded_force.addException(p1, p2, chargeProd, sigma, epsilon)

        self.system.addForce(standard_nonbonded_force)

    def add_custom_solute_nb_forces(self, solvent, solute):
        """
        Create CustomNonbondedForce to capture solute-solute and solute-solvent interactions.
        Assumes PME is in use.

        Taken from:
        https://github.com/openmm/openmm/pull/2014
        """

        nonbonded_force = self.system_forces["NonbondedForce"]

        # Determine PME parameters from nonbonded_force
        cutoff_distance = nonbonded_force.getCutoffDistance()
        [alpha_ewald, nx, ny, nz] = nonbonded_force.getPMEParameters()
        if (alpha_ewald / alpha_ewald.unit) == 0.0:
            # If alpha is 0.0, alpha_ewald is computed by OpenMM from from the error tolerance
            tol = nonbonded_force.getEwaldErrorTolerance()
            alpha_ewald = (1.0 / cutoff_distance) * math.sqrt(-math.log(2.0 * tol))
        # print(alpha_ewald)

        # Create CustomNonbondedForce
        ONE_4PI_EPS0 = 138.935456
        energy_expression = "((4*epsilon*((sigma/r)^12 - (sigma/r)^6)"
        energy_expression += "+ ONE_4PI_EPS0*chargeprod*erfc(alpha_ewald*r)/r));"
        energy_expression += "epsilon = sqrt(epsilon1*epsilon2);"
        energy_expression += "sigma = 0.5*(sigma1+sigma2);"
        energy_expression += "ONE_4PI_EPS0 = {:f};".format(
            ONE_4PI_EPS0
        )  # already in OpenMM units
        energy_expression += "chargeprod = charge1*charge2;"
        energy_expression += "alpha_ewald = {:f};".format(
            alpha_ewald.value_in_unit_system(unit.md_unit_system)
        )
        custom_nonbonded_force = openmm.CustomNonbondedForce(energy_expression)
        custom_nonbonded_force.addPerParticleParameter("charge")
        custom_nonbonded_force.addPerParticleParameter("sigma")
        custom_nonbonded_force.addPerParticleParameter("epsilon")
        # custom_nonbonded_force.addPerParticleParameter('soluteFlag')
        custom_nonbonded_force.addInteractionGroup(solute, solvent)

        # Configure force
        custom_nonbonded_force.setNonbondedMethod(
            openmm.CustomNonbondedForce.CutoffPeriodic
        )
        custom_nonbonded_force.setCutoffDistance(cutoff_distance)
        custom_nonbonded_force.setUseLongRangeCorrection(False)  # Not correct for LJ

        switch_flag = nonbonded_force.getUseSwitchingFunction()
        print("switch", switch_flag)
        if switch_flag:
            custom_nonbonded_force.setUseSwitchingFunction(True)
            switching_distance = nonbonded_force.getSwitchingDistance()
            custom_nonbonded_force.setSwitchingDistance(switching_distance)
        else:  # Truncated
            custom_nonbonded_force.setUseSwitchingFunction(False)

        # Copy particles
        self.init_nb_param = []
        for particle_index in range(nonbonded_force.getNumParticles()):
            [charge, sigma, epsilon] = nonbonded_force.getParticleParameters(
                particle_index
            )
            # solute_type = 1 if index in solute else 0
            # solute_type = 1
            # custom_nonbonded_force.addParticle([charge, sigma, epsilon, solute_type])
            custom_nonbonded_force.addParticle([charge, sigma, epsilon])
            self.init_nb_param.append([charge, sigma, epsilon])

        if solvent == solute:

            # Create CustomBondForce for exceptions
            energy_expression = "((4*epsilon*((sigma/r)^12 - (sigma/r)^6)"
            energy_expression += " + ONE_4PI_EPS0*chargeprod/r));"
            energy_expression += "ONE_4PI_EPS0 = {:f};".format(
                ONE_4PI_EPS0
            )  # already in OpenMM units
            custom_bond_force = openmm.CustomBondForce(energy_expression)
            custom_bond_force.addPerBondParameter("chargeprod")
            custom_bond_force.addPerBondParameter("sigma")
            custom_bond_force.addPerBondParameter("epsilon")

        # Copy solute-solute exclusions
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
            if iatom in solute and jatom in solute:
                self.init_nb_exept_index.append(exception_index)
                self.init_nb_exept_value.append(
                    [iatom, jatom, chargeprod, sigma, epsilon]
                )
            custom_nonbonded_force.addExclusion(iatom, jatom)
            if solvent == solute:

                if (
                    iatom in solute and jatom in solute
                ):  # or (iatom in solvent and jatom in solvent):
                    # print(iatom, jatom, chargeprod, sigma, epsilon)
                    custom_bond_force.addBond(
                        iatom, jatom, [chargeprod, sigma, epsilon]
                    )
                    if chargeprod._value != 0.0 and epsilon._value != 0.0:
                        print(iatom, jatom, chargeprod, sigma, epsilon)
                """
                if (iatom in solute and jatom in solute):
                    print("solute", iatom, jatom, chargeprod, sigma, epsilon)
                if (iatom in solute and jatom in solvent):
                    print("solute-solvent", iatom, jatom, chargeprod, sigma, epsilon)
                #if (iatom in solvent and jatom in solvent):
                #    print("Solvent", chargeprod, sigma, epsilon)
                """

        self.system.addForce(custom_nonbonded_force)
        self.system_forces["CustomNonbondedForce"] = custom_nonbonded_force
        self.add_negative_force(
            custom_nonbonded_force, name="NegativeCustomNonbondedForce"
        )

        if solvent == solute:

            self.system.addForce(custom_bond_force)
            self.add_negative_force(custom_bond_force)

    def update_custom_nonbonded(self, scale):

        custom_nonbonded_force = self.system_forces["CustomNonbondedForce"]

        for i in self.solute_index:
            q, sigma, eps = self.init_nb_param[i]
            custom_nonbonded_force.setParticleParameters(
                i, [q * math.sqrt(scale), sigma, eps * scale]
            )

        # Need to fix simulation
        custom_nonbonded_force.updateParametersInContext(self.simulation.context)

        custom_nonbonded_force = self.system_forces["NegativeCustomNonbondedForce"]

        for i in self.solute_index:
            q, sigma, eps = self.init_nb_param[i]
            custom_nonbonded_force.setParticleParameters(
                i, [q * math.sqrt(scale), sigma, eps * scale]
            )

        # Need to fix simulation
        custom_nonbonded_force.updateParametersInContext(self.simulation.context)


def print_forces(system, simulation):

    forces_dict = {}
    tot_ener = 0 * unit.kilojoules_per_mole

    for i, force in enumerate(system.getForces()):
        state = simulation.context.getState(getEnergy=True, groups={i})
        name = force.getName()
        pot_e = state.getPotentialEnergy()
        print(f"{force.getForceGroup():<3} {name:<25} {pot_e}")

        forces_dict[force.getForceGroup()] = {"name": name, "energy": pot_e}
        tot_ener += pot_e

    print(f'{len(forces_dict)+1:<3} {"Total":<25} {tot_ener}')

    forces_dict[len(forces_dict) + 1] = {"name": "Total", "energy": tot_ener}

    return forces_dict


def get_forces(system, simulation):

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


if __name__ == "__main__":

    # Check energy decompostion is correct:

    # Whole system:
    OUT_PATH = "/mnt/Data_3/SST2_clean/tmp"
    name = "2HPL"

    forcefield = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")

    dt = 2 * unit.femtosecond
    temperature = 300 * unit.kelvin
    friction = 1 / unit.picoseconds

    # SYSTEM

    pdb = PDBFile(f"{name}_equi_water.pdb")

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
    forces_sys = print_forces(system, simulation)

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

    pdb_pep = PDBFile(f"{name}_only_pep.pdb")

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
    forces_pep = print_forces(system_pep, simulation_pep)

    # NO Peptide system

    pdb_no_pep = PDBFile(f"{name}_no_pep.pdb")

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
    forces_no_pep = print_forces(system_no_pep, simulation_no_pep)

    ####################
    # ## REST2 test ####
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

    test = REST2(system, pdb, forcefield, solute_indices, integrator_rest)

    print("REST2 forces 300K:")
    forces_rest2 = print_forces(test.system, test.simulation)

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
        forces_rest2[4]["energy"]
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
    print(f"PeriodicTorsionForce {torsion_force/forces_pep[3]['energy']:.5e}")

    print("\nCompare torsion energy rest2 vs. no pep:\n")
    torsion_force = forces_rest2[6]["energy"]
    print(f"PeriodicTorsionForce {torsion_force/forces_no_pep[3]['energy']:.5e}")

    print("\nCompare nonbond energy rest2 vs. no pep+pep+solvent_solute_nb:\n")
    non_bonded = (
        solvent_solute_nb + forces_pep[2]["energy"] + forces_no_pep[2]["energy"]
    )
    print(f"NonbondedForce       {non_bonded/forces_sys[2]['energy']:.5e}")

    solute_scaled_force = forces_rest2[4]["energy"] + forces_pep[2]["energy"]
    print(f"E_solute_scaled      {solute_scaled_force/E_solute_scaled:.5e}")
    solute_not_scaled_force = (
        forces_rest2[5]["energy"] + forces_pep[0]["energy"] + +forces_pep[1]["energy"]
    )
    print(f"E_solute_not_scaled  {non_bonded/forces_sys[2]['energy']:.5e}")

    print(f"E_solvent            {E_solvent/forces_no_pep[6]['energy']:.5e}")

    scale = 0.5
    test.scale_nonbonded_torsion(scale)
    print("REST2 forces 600K:")
    forces_rest2 = print_forces(test.system, test.simulation)
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
    print(f"PeriodicTorsionForce {torsion_force/forces_pep[3]['energy']:.5e}")

    print("\nCompare torsion energy rest2 vs. no pep:\n")
    torsion_force = forces_rest2[6]["energy"]
    print(f"PeriodicTorsionForce {torsion_force/forces_no_pep[3]['energy']:.5e}")

    print("\nCompare nonbond energy rest2 vs. no pep+pep+solvent_solute_nb:\n")
    non_bonded = (
        solvent_solute_nb + forces_pep[2]["energy"] + forces_no_pep[2]["energy"]
    )
    print(f"NonbondedForce       {non_bonded/forces_sys[2]['energy']:.5e}")

    solute_scaled_force = forces_rest2[4]["energy"] + forces_pep[2]["energy"]
    print(f"E_solute_scaled      {solute_scaled_force/E_solute_scaled:.5e}")
    solute_not_scaled_force = (
        forces_rest2[5]["energy"] + forces_pep[0]["energy"] + +forces_pep[1]["energy"]
    )
    print(f"E_solute_not_scaled  {non_bonded/forces_sys[2]['energy']:.5e}")

    print(f"E_solvent            {E_solvent/forces_no_pep[6]['energy']:.5e}")

    """
    print('Timing %d steps of integration...' % nsteps)

    temp_sim = 450

    out_name = f'tmp/test_{temp_sim:d}K'
    tot_steps = 500000
    save_step_dcd = 10000
    save_step_log = 500

    scale = 300 / temp_sim

    test.simulation.reporters.append(
        DCDReporter(out_name +'_sim_temp.dcd',
                    save_step_dcd))

    test.simulation.reporters.append(
        StateDataReporter(sys.stdout, save_step_dcd,
                          step=True,
                          temperature=True,
                          speed=True,
                          remainingTime=True,
                          totalSteps=tot_steps))

    test.simulation.reporters.append(
        StateDataReporter(out_name +'_sim_temp.csv', 
                          save_step_log,
                          step=True,
                          potentialEnergy=True,
                          totalEnergy=True,
                          speed=True,
                          temperature=True))

    test.simulation.reporters.append(
        CheckpointReporter(
            out_name +'_sim_temp.chk',
            100000))

    class Rest2Reporter(object):
        def __init__(self, file, reportInterval, rest2):
            self._out = open(file, 'w')
            self._out.write("ps,Solute(kJ/mol),Solvent(kJ/mol),Solute-Solvent(kJ/mol)\n")
            self._reportInterval = reportInterval
            self._rest2 = rest2
            
        def __del__(self):
            self._out.close()

        def describeNextReport(self, simulation):
            steps = self._reportInterval - simulation.currentStep%self._reportInterval
            return (steps, False, False, False, False, None)

        def report(self, simulation, state):

            energies = self._rest2.compute_all_energies()

            time = state.getTime().value_in_unit(unit.picosecond)
            self._out.write(f'{time},{energies[0]._value},{energies[1]._value},{energies[2]._value}\n') 

    test.simulation.reporters.append(
        Rest2Reporter(
            out_name +'_energie_sim_temp.csv',
            500,
            test))

    # At 500K (βm/β0)
    # T0/Tm


    set_coor_pep = True
    nsteps = tot_steps
    every_step = 1000

    test.scale_nonbonded_torsion(scale)

    initial_time = time.time()
    for i in range(nsteps//every_step):
        #print('.', end='')
        test.simulation.step(every_step)
        
        #_update_custom_nonbonded(simulation, scale, solute_solvent_custom_nonbonded_force, init_nb_param)
        
        if set_coor_pep:
            #solute_force = test.compute_solute_energy()
            #solvent_force = test.compute_solvent_energy()

            #for i, force in solute_force.items():
            #    if force['name'] == 'NonbondedForce':
            #        solute_nb = force['energy']._value

            #for i, force in solvent_force.items():
            #    if force['name'] == 'NonbondedForce':
            #        solvent_nb = force['energy']._value

            #print(f"solute nonbonde = {solute_nb:>10.2f} solvent nonbonde = {solvent_nb:>10.2f}")

            #test.scale_nonbonded_torsion(scale)
            test.compute_all_energies()
            #test.scale_nonbonded_torsion(scale)
            #test.update_nonbonded(scale)
            #test.update_custom_nonbonded(scale)
            #test.update_torsions(scale)

            #test.scale_nonbonded_torsion(scale)
            #tot_positions = test.simulation.context.getState(
            #    getPositions=True,
            #    getEnergy=True).getPositions()
            #simulation_pep.context.setPositions(tot_positions[solute_indices[0]:solute_indices[-1]+1])
            #forces = get_forces(system_pep, simulation_pep)

    print()
    print(scale)
    final_time = time.time()
    elapsed_time = (final_time - initial_time) * unit.seconds
    integrated_time = nsteps * dt
    print(f'{nsteps} steps of {dt/unit.femtoseconds:.1f} fs timestep' +
          f'({integrated_time/unit.picoseconds:.3f} ps) took {elapsed_time/unit.seconds:.3f}'+
          f' s : {(integrated_time/elapsed_time)/(unit.nanoseconds/unit.day):.3f} ns/day')

    forces_rest2 = get_forces(test.system, test.simulation)

    for key, val in forces_rest2.items():
        print(f"{key:3} {val['name']:20} {val['energy']}")


    # Classic
    # 10000 steps of 2.0 fs timestep(20.000 ps) took 4.925 s : 350.893 ns/day
    # With separate Bonds, Angles, Torsions and 
    # specific Non bonded Solvent-Solute
    # 10000 steps of 2.0 fs timestep(20.000 ps) took 5.258 s : 328.624 ns/day
    # With separate Bonds, Angles, Torsions and 
    # specific Non bonded Solute-Solute and Solvent-Solute
    # 10000 steps of 2.0 fs timestep(20.000 ps) took 5.698 s : 303.262 ns/day

    """

    """
    vmd test_2HPL/2HPL_em_water.pdb test_2HPL/2HPL_equi_water.dcd -m 2HPL.pdb
    pbc wrap -molid 0 -first 0 -last last -compound fragment -center com -centersel "chain A and protein" -orthorhombic
    """
