#!/usr/bin/env python3
# coding: utf-8


"""
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

__author__ = "Samuel Murail"
__version__ = "0.0.1"

import openmm.unit as unit
import random
import os
from sys import stdout
import pandas as pd
import numpy as np
import logging

# Logging
logger = logging.getLogger(__name__)

from SST2.rest1 import run_rest1


class SST1Reporter(object):
    def __init__(self, sst1):
        self.sst1 = sst1

    def describeNextReport(self, simulation):
        steps1 = (
            self.sst1.tempChangeInterval
            - simulation.currentStep % self.sst1.tempChangeInterval
        )
        steps2 = (
            self.sst1.reportInterval - simulation.currentStep % self.sst1.reportInterval
        )
        steps = min(steps1, steps2)
        isUpdateAttempt = steps1 == steps
        return (steps, False, isUpdateAttempt, False, isUpdateAttempt)

    def report(self, simulation, state):
        energie_group = self.sst1.rest1.compute_all_energies()
        # energie = st.rest1.get_customPotEnergie()
        # E = Bi*Epp + (B0Bi)**0.5 Epw
        # print(energie_group)
        # energie = self.sst1.inverseTemperatures[self.sst1.currentTemperature] * energie_group[0] +\
        #    (self.sst1.inverseTemperatures[self.sst1.currentTemperature]*self.sst1.inverseTemperatures[0])**0.5 * energie_group[2]
        # energie *= unit.kilojoule / unit.mole
        # print('Energie', energie)
        self.sst1._e_num[self.sst1.currentTemperature] += 1
        self.sst1._e_solute_avg[self.sst1.currentTemperature] += (
            energie_group[0] - self.sst1._e_solute_avg[self.sst1.currentTemperature]
        ) / self.sst1._e_num[self.sst1.currentTemperature]
        self.sst1._e_solute_solv_avg[self.sst1.currentTemperature] += (
            energie_group[2]
            - self.sst1._e_solute_solv_avg[self.sst1.currentTemperature]
        ) / self.sst1._e_num[self.sst1.currentTemperature]
        # print(energie_group[0], energie_group[3])
        # print([ener._value for ener in st._e_solute_avg])
        # print([ener._value for ener in st._e_solute_solv_avg])
        if simulation.currentStep % self.sst1.reportInterval == 0:
            self.sst1._writeReport(energie_group)
        if simulation.currentStep % self.sst1.tempChangeInterval == 0:
            self.sst1._attemptTemperatureChange(state, energie_group[0], energie_group[2])


class SST1(object):
    """SimulatedTempering1 implements the simulated tempering algorithm for
    accelerated sampling.

    The set of temperatures to sample can be specified in two ways.  First,
    you can explicitly provide a list
    of temperatures by using the "temperatures" argument.  Alternatively,
    you can specify the minimum and
    maximum temperatures, and the total number of temperatures to use.
    The temperatures are chosen spaced
    exponentially between the two extremes.  For example,

    st = SimulatedTempering(simulation, numTemperatures=15,
                            minTemperature=300*kelvin,
                            maxTemperature=450*kelvin)

    After creating the SimulatedTempering object, call step() on it to
    run the simulation.

    Transitions between temperatures are performed at regular intervals,
    as specified by the "tempChangeInterval" argument.  For each transition,
    a new temperature is selected using the independence sampling method, as
    described in Chodera, J. and Shirts, M., J. Chem. Phys. 135, 194110
    (2011).

    Simulated tempering requires a "weight factor" for each temperature.
    Ideally, these should be chosen so
    the simulation spends equal time at every temperature.  You can specify
    the list of weights to use with the optional "weights" argument.  If
    this is omitted, weights are selected automatically using the Wang-Landau
    algorithm as described in Wang, F. and Landau, D. P., Phys. Rev. Lett.
    86(10), pp. 2050-2053 (2001).

    To properly analyze the results of the simulation, it is important
    to know the temperature and weight factors at every point in time.
    The SimulatedTempering object functions as a reporter, writing this
    information to a file or stdout at regular intervals (which should
    match the interval at which you save frames from the simulation).
    You can specify the output file and reporting interval with the
    "reportFile" and "reportInterval" arguments.

    Parameters
    ----------
    rest1: REST1
        The REST1 object defining the System, Context, and Integrator to use
    simulation: Simulation
        The Simulation defining the System, Context, and Integrator to use

    Methods
    -------
    step(steps)
        Run a number of time steps.
    """

    def __init__(
        self,
        rest1,
        temperatures,
        refTemperature=None,
        weights=None,
        tempChangeInterval=25,
        reportInterval=1000,
        reportFile=stdout,
        restart_files=None,
        restart_files_full=None,
    ):
        """Create a new SimulatedTempering.

        Parameters
        ----------
        simulation: Simulation
            The Simulation defining the System, Context, and Integrator to use
        temperatures: list
            The list of temperatures to use for tempering, in increasing order
        refTemperature: temperature
            The reference temperature to use for tempering. If this is not specified, the first temperature in the list is used.
        weights: list
            The weight factor for each temperature.  If none, weights are selected automatically.
        tempChangeInterval: int
            The interval (in time steps) at which to attempt transitions between temperatures
        reportInterval: int
            The interval (in time steps) at which to write information to the report file
        reportFile: string or file
            The file to write reporting information to, specified as a file name or file object
        restart_files: list of strings
            Files to read restart information to, specified as a file name
        restart_files_full: string
            Full Rest1 files to read restart information to, specified as a file name
        """
        self.rest1 = rest1
        self.simulation = rest1.simulation

        numTemperatures = len(temperatures)
        self.temperatures = [
            t.in_units_of(unit.kelvin) if unit.is_quantity(t) else t * unit.kelvin
            for t in temperatures
        ]
        minTemperature = self.temperatures[0]
        maxTemperature = self.temperatures[-1]

        if refTemperature is None:
            self.refTemperature = minTemperature
        else:
            if unit.is_quantity(refTemperature):
                self.refTemperature = refTemperature.in_units_of(unit.kelvin)
            else:
                self.refTemperature = refTemperature * unit.kelvin

        assert (
            self.refTemperature in self.temperatures
        ), f"Reference temperature {self.refTemperature} not in temperatures_list {self.temperatures}"
        self.temp_ref_index = self.temperatures.index(self.refTemperature)

        if any(
            self.temperatures[i] >= self.temperatures[i + 1]
            for i in range(numTemperatures - 1)
        ):
            raise ValueError("The temperatures must be in strictly increasing order")

        logger.info(
            f"Min={minTemperature}, Ref={refTemperature}, Max={maxTemperature}, temp_list={[temp._value for temp in self.temperatures]}"
        )
        self.tempChangeInterval = tempChangeInterval
        self.reportInterval = reportInterval
        self.inverseTemperatures = [
            1.0 / (unit.MOLAR_GAS_CONSTANT_R * t) for t in self.temperatures
        ]

        # If necessary, open the file we will write reports to.

        self._openedFile = isinstance(reportFile, str)
        if self._openedFile:
            self._out = open(reportFile, "w", 1)
        else:
            self._out = reportFile

        # Initialize the weights.

        if weights is None:
            first_temp_index = self.compute_starting_weight(
                restart_files, restart_files_full
            )
            self._updateWeights = True
        else:
            self._weights = weights
            self._updateWeights = False

        # Select the initial temperature.
        if restart_files is None:
            self.currentTemperature = 0
        elif weights is None:
            self.currentTemperature = first_temp_index
        else:
            # Need to treat the case where weights is not None and restart_files is not None
            # TO CHANGE ! This is BAD MOKAY !!!!! :
            self.currentTemperature = 0

        # print(self.temperatures[self.currentTemperature])
        # self.simulation.integrator.setTemperature(self.temperatures[self.currentTemperature])
        self.rest1.scale_nonbonded_bonded(
            self.temperatures[self.temp_ref_index]
            / self.temperatures[self.currentTemperature]
        )

        self.simulation.integrator.setTemperature(self.temperatures[self.currentTemperature])

        # Add a reporter to the simulation which will handle the updates and reports.
        self.simulation.reporters.append(SST1Reporter(self))

        # Write out the header line.

        headers = [
            "Step",
            "Aim Temp (K)",
            "E solute (kJ/mole)",
            "E solvent (kJ/mole)",
            "E solvent-solute (kJ/mole)",
        ]
        print((",").join(headers), file=self._out)

    def compute_starting_weight(self, restart_files, restart_files_full):
        """Compute the weight factor for each temperature.

        Parameters
        ----------
        restart_files: list of strings
            Files to read restart information to, specified as a file name
        restart_files_full: string
            Full Rest1 files to read restart information to, specified as a file name

        Returns
        -------
        first_temp_index: int
            Index of the last used temperature to use
        """
        numTemperatures = len(self.temperatures)
        # Initialize the energy arrays.
        self._e_num = [0] * numTemperatures
        self._e_solute_avg = [0.0 * unit.kilojoules_per_mole] * numTemperatures
        self._e_solute_solv_avg = [0.0 * unit.kilojoules_per_mole] * numTemperatures
        self._weights = [0.0] * numTemperatures

        # For restart, weight should be recomputed based on previous results
        if restart_files is not None and restart_files_full is not None:
            df_sim = pd.read_csv(restart_files[0])
            df_temp = pd.read_csv(restart_files_full[0])

            for i in range(1, len(restart_files)):
                logger.info(f"Reading part {i}")
                df_sim_part = pd.read_csv(restart_files[i])
                df_temp_part = pd.read_csv(restart_files_full[i])

                df_sim = (
                    pd.concat([df_sim, df_sim_part], axis=0, join="outer")
                    .reset_index()
                    .drop(["index"], axis=1)
                )
                df_temp = (
                    pd.concat([df_temp, df_temp_part], axis=0, join="outer")
                    .reset_index()
                    .drop(["index"], axis=1)
                )

            # Remove Nan rows (rare cases of crashes)
            df_sim = df_sim[df_sim.iloc[:, 0].notna()]
            df_sim["Temperature (K)"] = df_temp["Aim Temp (K)"]
            temp_array = df_sim["Temperature (K)"].unique()
            temp_array.sort()
            logger.info(temp_array)

            # Remove Nan rows (rare cases of crashes)
            df_temp = df_temp[df_temp.iloc[:, 0].notna()]

            for temp_index, temp in enumerate(temp_array):
                df_local = df_temp[df_temp["Aim Temp (K)"] == temp]
                self._e_num[temp_index] = len(df_local)
                self._e_solute_avg[temp_index] = (
                    df_local["E solute scaled (kJ/mole)"].mean()
                    * unit.kilojoules_per_mole
                )
                self._e_solute_solv_avg[temp_index] = (
                    df_local["E solvent-solute (kJ/mole)"].mean()
                    * unit.kilojoules_per_mole
                )

            first_temp_index = 0
            for index, row in df_sim.iloc[::-1].iterrows():
                if index % (50 * 10) == 0:
                    temp_index = np.where(temp_array == row["Temperature (K)"])[0][0]
                    first_temp_index = temp_index
                    break

            logger.info(self._e_num)
            logger.info(self._e_solute_avg)
            logger.info(self._e_solute_solv_avg)
            logger.info(f"last temperature = {temp_array[first_temp_index]}")
            return first_temp_index
        else:
            return 0

    def _writeReport(self, energie_group):
        """Write out a line to the report."""
        temperature = self.temperatures[self.currentTemperature].value_in_unit(
            unit.kelvin
        )
        values = [temperature] + [energie._value for energie in energie_group]
        print(
            ("%d," % self.simulation.currentStep) + ",".join("%g" % v for v in values),
            file=self._out,
        )

    def __del__(self):
        if self._openedFile:
            self._out.close()

    @property
    def weights(self):
        return [x - self._weights[0] for x in self._weights]

    def step(self, steps):
        """Advance the simulation by integrating a specified number of time steps."""
        self.simulation.step(steps)

    def _compute_weight(self, i, j):
        r"""Compute the difference of weight $w_j - w_i$
        using the following equation:

        $$(w_j - w_i) = (\beta_j - \beta_i) \frac{ (\braket{E_{pp}^{(1)}}_i -  \braket{E_{pp}^{(1)}}_j)}{2} +  (\sqrt{\beta_{ref} \beta_j} - \sqrt{\beta_{ref} \beta_i}) \frac {(\braket{E_{pw}}_i - \braket{E_{pw}}_j)}{2}$$
        """

        if self._e_num[j] != 0:
            avg_ener_solut = self._e_solute_avg[i] / 2
            avg_ener_solut += self._e_solute_avg[j] / 2
            avg_ener_solut_solv = self._e_solute_solv_avg[i] / 2
            avg_ener_solut_solv += self._e_solute_solv_avg[j] / 2

        else:
            avg_ener_solut = self._e_solute_avg[i]
            avg_ener_solut_solv = self._e_solute_solv_avg[i]

        weight = (
            self.inverseTemperatures[j] - self.inverseTemperatures[i]
        ) * avg_ener_solut
        weight += avg_ener_solut_solv * (
            (
                self.inverseTemperatures[j]
                * self.inverseTemperatures[self.temp_ref_index]
            )
            ** 0.5
            - (
                self.inverseTemperatures[i]
                * self.inverseTemperatures[self.temp_ref_index]
            )
            ** 0.5
        )

        return weight

    def _attemptTemperatureChange(self, state, ener_solut, ener_solut_solv):
        """Attempt to move to a different temperature."""

        temp_list = []

        temp_i = self.currentTemperature

        if self.currentTemperature != 0:
            temp_list.append(temp_i - 1)
        if self.currentTemperature < (len(self._weights) - 1):
            temp_list.append(temp_i + 1)

        logProbability = []
        # Compute Delta_(i,j) = (Bi-Bj)Epp + ((BrefBi)**0.5 - (BrefBj)**0.5)Epw - (fi-fj)
        for j in temp_list:
            log_prob = (
                self.inverseTemperatures[temp_i] - self.inverseTemperatures[j]
            ) * ener_solut
            log_prob += (
                (
                    self.inverseTemperatures[temp_i]
                    * self.inverseTemperatures[self.temp_ref_index]
                )
                ** 0.5
                - (
                    self.inverseTemperatures[j]
                    * self.inverseTemperatures[self.temp_ref_index]
                )
                ** 0.5
            ) * ener_solut_solv
            weight = self._compute_weight(temp_i, j)
            log_prob += weight
            logProbability.append(log_prob)

        probability = [np.exp(x) for x in logProbability]

        # To avoid trying always i-1 in first
        # add a random on which temp index to test first.
        # Might need to compute the combinatory of p(i-1), p(i+1)
        # to compute p(i)
        index_list = list(range(len(temp_list)))
        random.shuffle(index_list)

        for i in index_list:
            r = random.random()

            if r < probability[i]:
                # print(f"SWITCH {self.currentTemperature:2} -> {temp_list[i]:2}")
                # Select the new temperature.
                self.currentTemperature = temp_list[i]
                # self.simulation.integrator.setTemperature(self.temperatures[i])
                self.rest1.scale_nonbonded_bonded(
                    self.temperatures[self.temp_ref_index]
                    / self.temperatures[temp_list[i]]
                )

                scale = (
                    self.temperatures[self.temp_ref_index]
                        / self.temperatures[self.currentTemperature]
                ) ** 0.5
                velocities = scale * state.getVelocities(asNumpy=True).value_in_unit(
                    unit.nanometers / unit.picoseconds
                )
                self.simulation.context.setVelocities(velocities)
                self.simulation.integrator.setTemperature(self.temperatures[self.currentTemperature])

                break


def run_sst1(
    sys_rest1,
    generic_name,
    tot_steps,
    dt,
    temperatures,
    ref_temp,
    save_step_dcd=100000,
    save_step_log=500,
    tempChangeInterval=500,
    reportInterval=500,
    overwrite=False,
    save_checkpoint_steps=None,
):
    """
    Run a SST1 simulation.

    Parameters
    ----------
    sys_rest1 : Rest1 object
        The system to simulate.
    generic_name : str
        Generic name for the output files.
    tot_steps : int
        Total number of steps to run.
    dt : float
        Time step in fs.
    temperatures : list of float
        List of temperatures to simulate.
    ref_temp : float
        Reference temperature.
    save_step_dcd : int, optional
        Number of steps between each DCD save. The default is 100000.
    save_step_log : int, optional
        Number of steps between each log save. The default is 500.
    tempChangeInterval : int, optional
        Number of steps between each temperature change. The default is 500.
    reportInterval : int, optional
        Number of steps between each report. The default is 500.
    overwrite : bool, optional
        Overwrite the previous simulation. The default is True.
    save_checkpoint_steps : int, optional
        Number of steps between each checkpoint save. The default is None.


    """

    if unit.is_quantity(ref_temp):
        ref_temp = ref_temp.in_units_of(unit.kelvin)
    else:
        ref_temp *= unit.kelvin

    assert (
        ref_temp in temperatures
    ), f"Reference temperature {ref_temp} not in temperatures_list {temperatures}"

    report_sst1 = f"{generic_name}_sst1_full.csv"
    restart_files = None
    restart_files_full = None
    tot_steps = np.ceil(tot_steps)

    if not overwrite and os.path.isfile(report_sst1):
        logger.info(
            f"File {generic_name}_sst1_full.csv exists already, restart run_sst1() step"
        )
        # Get part number
        part = 2

        report_sst1 = f"{generic_name}_sst1_full_part_{part}.csv"
        report_simple_sst1 = f"{generic_name}_sst1_part_{part}.csv"

        restart_files = [f"{generic_name}_sst1.csv"]
        restart_files_full = [f"{generic_name}_sst1_full.csv"]

        while os.path.isfile(report_sst1):
            restart_files.append(report_simple_sst1)
            restart_files_full.append(report_sst1)
            report_sst1 = f"{generic_name}_sst1_full_part_{part}.csv"
            report_simple_sst1 = f"{generic_name}_sst1_part_{part}.csv"
            part += 1

        if part != 2:
            restart_files = restart_files[:-1]
            restart_files_full = restart_files_full[:-1]

        logger.info(f"Using restart file : {restart_files}")

    sys_rest1.simulation.reporters = []
    sys_rest1.simulation.currentStep = 0

    sst1 = SST1(
        sys_rest1,
        temperatures=temperatures,
        refTemperature=ref_temp,
        tempChangeInterval=tempChangeInterval,
        reportFile=report_sst1,
        reportInterval=reportInterval,
        restart_files=restart_files,
        restart_files_full=restart_files_full,
    )

    logger.info(f"- Launch SST1")
    run_rest1(
        sst1.rest1,
        f"{generic_name}_sst1",
        tot_steps=tot_steps,
        dt=dt,
        save_step_dcd=save_step_dcd,
        save_step_log=save_step_log,
        save_step_rest1=reportInterval,
        add_REST1_reporter=False,
        remove_reporters=False,
        save_checkpoint_steps=save_checkpoint_steps,
    )
