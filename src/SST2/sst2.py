#!/usr/bin/env python3
# coding: utf-8


"""
simulatedtempering.py: Implements simulated tempering

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2015 Stanford University and the Authors.
Authors: Peter Eastman
Contributors:

Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
__author__ = "Peter Eastman"
__version__ = "1.0"

import openmm.unit as unit
import random
import os
from sys import stdout
import pandas as pd
import numpy as np
import logging

try:
    import bz2

    have_bz2 = True
except:
    have_bz2 = False

try:
    import gzip

    have_gzip = True
except:
    have_gzip = False

# Logging
logger = logging.getLogger(__name__)

from SST2.rest2 import run_rest2


class SST2Reporter(object):
    def __init__(self, sst2):
        self.sst2 = sst2
    def describeNextReport(self, simulation):
        sst2 = self.sst2
        steps1 = (
            sst2.tempChangeInterval
            - simulation.currentStep % sst2.tempChangeInterval
        )
        steps2 = (
            sst2.reportInterval - simulation.currentStep % sst2.reportInterval
        )
        steps = min(steps1, steps2)
        isUpdateAttempt = steps1 == steps
        return (steps, False, isUpdateAttempt, False, isUpdateAttempt)
    def report(self, simulation, state):
        st = self.sst2
        energie_group = st.rest2.compute_all_energies()
        # energie = st.rest2.get_customPotEnergie()
        # E = Bi*Epp + (B0Bi)**0.5 Epw
        # print(energie_group)
        # energie = self.sst2.inverseTemperatures[self.sst2.currentTemperature] * energie_group[0] +\
        #    (self.sst2.inverseTemperatures[self.sst2.currentTemperature]*self.sst2.inverseTemperatures[0])**0.5 * energie_group[2]
        # energie *= unit.kilojoule / unit.mole
        # print('Energie', energie)
        st._e_num[st.currentTemperature] += 1
        st._e_solute_avg[st.currentTemperature] += (
            energie_group[0] - st._e_solute_avg[st.currentTemperature]
        ) / st._e_num[st.currentTemperature]
        st._e_solute_solv_avg[st.currentTemperature] += (
            energie_group[3] - st._e_solute_solv_avg[st.currentTemperature]
        ) / st._e_num[st.currentTemperature]
        # print(energie_group[0], energie_group[3])
        # print([ener._value for ener in st._e_solute_avg])
        # print([ener._value for ener in st._e_solute_solv_avg])
        if simulation.currentStep % st.reportInterval == 0:
            st._writeReport(energie_group)
        if simulation.currentStep % st.tempChangeInterval == 0:
            st._attemptTemperatureChange(
                state, energie_group[0], energie_group[3]
            )

class SST2(object):
    """SimulatedTempering implements the simulated tempering algorithm for
    accelerated sampling.

    It runs a simulation while allowing the temperature to vary.  At high
    temperatures, it can more easily cross energy barriers to explore a wider
    area of conformation space.  At low temperatures, it can thoroughly
    explore each local region.  For details, see Marinari, E. and Parisi, G.,
    Europhys. Lett. 19(6). pp. 451-458 (1992).

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
    rest2: REST2
        The REST2 object defining the System, Context, and Integrator to use
    simulation: Simulation
        The Simulation defining the System, Context, and Integrator to use
    
    Methods
    -------
    step(steps)
        Run a number of time steps.
    """

    def __init__(
        self,
        rest2,
        temperatures,
        refTemperature=None,
        weights=None,
        tempChangeInterval=25,
        reportInterval=1000,
        reportFile=stdout,
        restart_file=None,
        restart_file_full=None,
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
        """
        self.rest2 = rest2
        self.simulation = rest2.simulation

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
            # Detect the desired compression scheme from the filename extension
            # and open all files unbuffered
            if reportFile.endswith(".gz"):
                if not have_gzip:
                    raise RuntimeError(
                        "Cannot write .gz file because Python could not import gzip library"
                    )
                self._out = gzip.GzipFile(fileobj=open(reportFile, "wb", 0))
            elif reportFile.endswith(".bz2"):
                if not have_bz2:
                    raise RuntimeError(
                        "Cannot write .bz2 file because Python could not import bz2 library"
                    )
                self._out = bz2.BZ2File(reportFile, "w", 0)
            else:
                self._out = open(reportFile, "w", 1)
        else:
            self._out = reportFile

        # Initialize the weights.

        if weights is None:
            self._e_num = [0] * numTemperatures
            self._e_solute_avg = [0.0 * unit.kilojoule / unit.mole] * numTemperatures
            self._e_solute_solv_avg = [
                0.0 * unit.kilojoule / unit.mole
            ] * numTemperatures
            self._weights = [0.0] * numTemperatures
            self._updateWeights = True

            # For restart, weight should be recomputed based on previous results
            if restart_file is not None and restart_file_full is not None:
                df_sim = pd.read_csv(restart_file[0])
                df_temp = pd.read_csv(restart_file_full[0])

                for i in range(1, len(restart_file)):
                    logger.info(f"Reading part {i}")
                    df_sim_part = pd.read_csv(restart_file[i])
                    df_temp_part = pd.read_csv(restart_file_full[i])

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
                        * unit.kilojoule
                        / unit.mole
                    )
                    self._e_solute_solv_avg[temp_index] = (
                        df_local["E solvent-solute (kJ/mole)"].mean()
                        * unit.kilojoule
                        / unit.mole
                    )

                first_temp_index = 0
                for index, row in df_sim.iloc[::-1].iterrows():
                    if index % (50 * 10) == 0:
                        temp_index = np.where(temp_array == row["Temperature (K)"])[0][
                            0
                        ]
                        first_temp_index = temp_index
                        break

                logger.info(self._e_num)
                logger.info(self._e_solute_avg)
                logger.info(self._e_solute_solv_avg)
                logger.info(f"last temperature = {temp_array[first_temp_index]}")

        else:
            self._weights = weights
            self._updateWeights = False

        # Select the initial temperature.

        if restart_file is None:
            self.currentTemperature = 0
        else:
            self.currentTemperature = first_temp_index
        # print(self.temperatures[self.currentTemperature])
        # self.simulation.integrator.setTemperature(self.temperatures[self.currentTemperature])
        self.rest2.scale_nonbonded_torsion(
            self.temperatures[self.temp_ref_index]
            / self.temperatures[self.currentTemperature]
        )
        # Add a reporter to the simulation which will handle the updates and reports.

        self.simulation.reporters.append(SST2Reporter(self))

        # Write out the header line.

        headers = [
            "Steps",
            "Aim Temp (K)",
            "E solute scaled (kJ/mole)",
            "E solute not scaled (kJ/mole)",
            "E solvent (kJ/mole)",
            "E solvent-solute (kJ/mole)",
        ]
        print('"%s"' % ('","').join(headers), file=self._out)

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

        # avg_ener_solut *= unit.kilojoule / unit.mole
        # avg_ener_solut_solv *= unit.kilojoule / unit.mole

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
        # print(temp_i, temp_list)
        for j in temp_list:
            log_prob = (
                self.inverseTemperatures[temp_i] - self.inverseTemperatures[j]
            ) * ener_solut
            # print('logprob 1 =', log_prob)
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
            # print('logprob 2 =', log_prob)
            weight = self._compute_weight(temp_i, j)
            log_prob += weight
            # print('logprob 3 =', log_prob, weight)
            logProbability.append(log_prob)
            # print(temp_i, j, log_prob)

        # print('Proba:', probability)
        probability = [np.exp(x) for x in logProbability]
        # print('New Proba:', probability)
        # print('Temp list', temp_list)

        # To avoid trying i-1 in first
        # I add a random on which temp index to test
        # First.
        # Might need to compute the combinatory of p(i-1), p(i+1)
        # to compute p(i)
        index_list = list(range(len(temp_list)))
        # print(index_list)
        random.shuffle(index_list)
        # print(index_list)

        for i in index_list:
            r = random.random()

            if r < probability[i]:
                # print(f"SWITCH {self.currentTemperature:2} -> {temp_list[i]:2}")
                # Rescale the velocities.
                # scale = math.sqrt(self.temperatures[i]/self.temperatures[self.currentTemperature])
                # if have_numpy:
                #     velocities = scale*state.getVelocities(asNumpy=True).value_in_unit(unit.nanometers/unit.picoseconds)
                # else:
                #     velocities = [v*scale for v in state.getVelocities().value_in_unit(unit.nanometers/unit.picoseconds)]
                # self.simulation.context.setVelocities(velocities)

                # Select this temperature.

                self.currentTemperature = temp_list[i]
                # self.simulation.integrator.setTemperature(self.temperatures[i])
                self.rest2.scale_nonbonded_torsion(
                    self.temperatures[self.temp_ref_index]
                    / self.temperatures[temp_list[i]]
                )

                break


# -651.2058710416085 kJ/mol -2082.3374353838976 kJ/mol
# [-774.0805157886525, -746.9156031589454, -745.8530421384646, -734.1762990186642, -711.7915332524436, -729.9946964283761, -694.0997403755065, -663.2442182707645, 0.0, 0.0]
# [-2437.0540461474134, -2362.545113976842, -2297.780046082346, -2252.782031692964, -2202.961638364378, -2094.6851449334276, -2016.3382915321304, -2031.4306585449015, 0.0, 0.0]


def run_sst2(
    sys_rest2,
    generic_name,
    tot_steps,
    dt,
    temperatures,
    ref_temp,
    save_step_dcd=100000,
    save_step_log=500,
    save_step_rest2=500,
    tempChangeInterval=500,
    reportInterval=500,
    overwrite=False,
    save_checkpoint_steps=None,
):
    """
    Run a SST2 simulation.

    Parameters
    ----------
    sys_rest2 : Rest2 object
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
    save_step_rest2 : int, optional
        Number of steps between each Rest2 save. The default is 500.
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

    report_sst2 = f"{generic_name}_sst2_full.csv"
    restart_file = None
    restart_file_full = None
    tot_steps = np.ceil(tot_steps)

    if not overwrite and os.path.isfile(report_sst2):
        logger.info(
            f"File {generic_name}_sst2_full.csv exists already, restart run_sst2() step"
        )
        # Get part number
        part = 2

        report_sst2 = f"{generic_name}_sst2_full_part_{part}.csv"
        report_simple_sst2 = f"{generic_name}_sst2_part_{part}.csv"

        restart_file = [f"{generic_name}_sst2.csv"]
        restart_file_full = [f"{generic_name}_sst2_full.csv"]

        while os.path.isfile(report_sst2):
            restart_file.append(report_simple_sst2)
            restart_file_full.append(report_sst2)
            report_sst2 = f"{generic_name}_sst2_full_part_{part}.csv"
            report_simple_sst2 = f"{generic_name}_sst2_part_{part}.csv"
            part += 1

        if part != 2:
            restart_file = restart_file[:-1]
            restart_file_full = restart_file_full[:-1]

        logger.info(f"Using restart file : {restart_file}")

    sys_rest2.simulation.reporters = []
    sys_rest2.simulation.currentStep = 0

    sst2 = SST2(
        sys_rest2,
        temperatures=temperatures,
        refTemperature=ref_temp,
        tempChangeInterval=tempChangeInterval,
        reportFile=report_sst2,
        reportInterval=reportInterval,
        restart_file=restart_file,
        restart_file_full=restart_file_full,
    )

    logger.info(f"- Launch SST2")
    run_rest2(
        sst2.rest2,
        f"{generic_name}_sst2",
        dt=dt,
        tot_steps=tot_steps,
        # topology=pdb.topology, # To fix !!!
        save_step_dcd=save_step_dcd,
        save_step_log=save_step_log,
        rest2_reporter=False,
        save_step_rest2=500,
        save_checkpoint_steps=save_checkpoint_steps,
    )


def compute_temperature_list(
    minTemperature, maxTemperature, numTemperatures, refTemperature=None
):
    """Compute the list of temperatures to simulate.

    Parameters
    ----------
    minTemperature : float
        Minimum temperature to simulate.
    maxTemperature : float
        Maximum temperature to simulate.
    numTemperatures : int
        Number of temperatures to simulate.
    refTemperature : float, optional
        Reference temperature. The default is None.

    """

    if unit.is_quantity(minTemperature):
        minTemperature = minTemperature.in_units_of(unit.kelvin)
    else:
        minTemperature *= unit.kelvin

    if unit.is_quantity(maxTemperature):
        maxTemperature = maxTemperature.in_units_of(unit.kelvin)
    else:
        maxTemperature *= unit.kelvin

    if refTemperature is not None:
        if unit.is_quantity(refTemperature):
            refTemperature = refTemperature.in_units_of(unit.kelvin)
        else:
            refTemperature *= unit.kelvin

    # Case with refTemp is minTemp
    temperatures = [
        minTemperature
        * ((maxTemperature / minTemperature) ** (i / float(numTemperatures - 1)))
        for i in range(numTemperatures)
    ]
    if refTemperature is None or refTemperature == minTemperature:
        refTemperature = minTemperature
    else:
        # Get closest temp to ref temp
        diff_temp = [abs(temp - refTemperature) for temp in temperatures]
        ref_index = diff_temp.index(min(diff_temp))

        if ref_index > 0:
            temperatures = [
                minTemperature * ((refTemperature / minTemperature) ** (i / ref_index))
                for i in range(ref_index)
            ]
            temperatures += [
                refTemperature
                * ((maxTemperature / refTemperature))
                ** (i / (numTemperatures - ref_index - 1))
                for i in range(numTemperatures - ref_index)
            ]
        else:
            temperatures = [minTemperature] + [
                refTemperature
                * ((maxTemperature / refTemperature)) ** (i / (numTemperatures - 2))
                for i in range(numTemperatures - 1)
            ]

    return temperatures
