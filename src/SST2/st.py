"""
Portions copyright (c) 2023 Université Paris Cité and the Authors.
Authors: Samuel Murail.

This package is largely inspired by the Peter Eastman's simulatedtempering
library from the OpenMM package. The original license is reproduced below :


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
__author__ = "Samuel Murail, Peter Eastman"
__version__ = "0.0.1"

import openmm.unit as unit
import math
import random
from sys import stdout
import pandas as pd
import numpy as np


class ST(object):
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
    """

    def __init__(
        self,
        simulation,
        temperatures=None,
        numTemperatures=None,
        minTemperature=None,
        maxTemperature=None,
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
        numTemperatures: int
            The number of temperatures to use for tempering.  If temperatures is not None, this is ignored.
        minTemperature: temperature
            The minimum temperature to use for tempering.  If temperatures is not None, this is ignored.
        maxTemperature: temperature
            The maximum temperature to use for tempering.  If temperatures is not None, this is ignored.
        weights: list
            The weight factor for each temperature.  If none, weights are selected automatically.
        tempChangeInterval: int
            The interval (in time steps) at which to attempt transitions between temperatures
        reportInterval: int
            The interval (in time steps) at which to write information to the report file
        reportFile: string or file
            The file to write reporting information to, specified as a file name or file object
        """
        self.simulation = simulation
        if temperatures is None:
            if unit.is_quantity(minTemperature):
                minTemperature = minTemperature.value_in_unit(unit.kelvin)
            if unit.is_quantity(maxTemperature):
                maxTemperature = maxTemperature.value_in_unit(unit.kelvin)
            self.temperatures = [
                minTemperature
                * (
                    (float(maxTemperature) / minTemperature)
                    ** (i / float(numTemperatures - 1))
                )
                for i in range(numTemperatures)
            ] * unit.kelvin
        else:
            numTemperatures = len(temperatures)
            self.temperatures = [
                (t.value_in_unit(unit.kelvin) if unit.is_quantity(t) else t)
                * unit.kelvin
                for t in temperatures
            ]
            if any(
                self.temperatures[i] >= self.temperatures[i + 1]
                for i in range(numTemperatures - 1)
            ):
                raise ValueError(
                    "The temperatures must be in strictly increasing order"
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
            self._e_pot_num = [0] * numTemperatures
            self._e_pot_sum = [0.0 * unit.kilojoule / unit.mole] * numTemperatures
            self._weights = [0.0] * numTemperatures
            self._updateWeights = True
            self._weights_factor = [1.0] * numTemperatures

            # For restart, weight should be recomputed based on previous results
            if restart_file is not None and restart_file_full is not None:
                df_sim = pd.read_csv(restart_file[0])
                df_temp = pd.read_csv(restart_file_full[0], sep="\t")

                for i in range(1, len(restart_file)):
                    print(f"Reading part {i}")
                    df_sim_part = pd.read_csv(restart_file[i])
                    df_temp_part = pd.read_csv(restart_file_full[i], sep="\t")

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

                df_sim["Temperature (K)"] = df_temp["Temperature (K)"]
                temp_array = df_temp["Temperature (K)"].unique()

                # In case df_sim and df_temp are not excactly the same length
                # Remove the NaN values
                df_sim = df_sim.dropna()

                print(temp_array)

                for index, row in df_sim.iterrows():
                    temp_index = np.where(temp_array == row["Temperature (K)"])[0][0]
                    self._e_pot_num[temp_index] += 1
                    self._e_pot_sum[temp_index] += (
                        row["Potential Energy (kJ/mole)"] * unit.kilojoule / unit.mole
                    )
                    if index % 50 == 0:
                        first_temp_index = temp_index

                print(f"last temperature = {temp_array[first_temp_index]}")

                for k in range(len(self._weights) - 1):
                    # Use Park and Pande weights:
                    # f(n+1) = fn + (β(n+1) − βn)*(E(n+1) + En)/2,

                    new_weight = (
                        self.inverseTemperatures[k + 1] - self.inverseTemperatures[k]
                    )
                    if self._e_pot_num[k + 1] != 0:
                        new_weight *= self._e_pot_sum[k] / (
                            2 * self._e_pot_num[k]
                        ) + self._e_pot_sum[k + 1] / (2 * self._e_pot_num[k + 1])
                        self._weights[k + 1] = (
                            self._weights[k] + new_weight * self._weights_factor[k + 1]
                        )
                        # self._weights[k + 1] *= self._weights_factor[k + 1]

                    # Use H. Nguyen hack
                    # f(n+1) = fn + (β(n+1) − βn)*En,
                    else:
                        new_weight *= self._e_pot_sum[k] / (self._e_pot_num[k])
                        self._weights[k + 1] = (
                            self._weights[k] + new_weight * self._weights_factor[k + 1]
                        )
                        # self._weights[k + 1] *= self._weights_factor[k + 1]
                        break

                print(self._weights)

        else:
            self._weights = weights
            self._updateWeights = False

        # Select the initial temperature.

        if restart_file is None:
            self.currentTemperature = 0
        else:
            self.currentTemperature = first_temp_index
        # print(self.temperatures[self.currentTemperature])
        self.simulation.integrator.setTemperature(
            self.temperatures[self.currentTemperature]
        )

        # Add a reporter to the simulation which will handle the updates and reports.

        class STReporter(object):
            def __init__(self, st):
                self.st = st

            def describeNextReport(self, simulation):
                st = self.st
                steps1 = (
                    st.tempChangeInterval
                    - simulation.currentStep % st.tempChangeInterval
                )
                steps2 = st.reportInterval - simulation.currentStep % st.reportInterval
                steps = min(steps1, steps2)
                isUpdateAttempt = steps1 == steps
                return (steps, False, isUpdateAttempt, False, isUpdateAttempt)

            def report(self, simulation, state):
                st = self.st

                st._e_pot_num[st.currentTemperature] += 1
                st._e_pot_sum[st.currentTemperature] += state.getPotentialEnergy()

                if simulation.currentStep % st.tempChangeInterval == 0:
                    st._attemptTemperatureChange(state)
                if simulation.currentStep % st.reportInterval == 0:
                    st._writeReport()

        simulation.reporters.append(STReporter(self))

        # Write out the header line.

        headers = ["Steps", "Temperature (K)"]
        for t in self.temperatures:
            headers.append("%gK Weight" % t.value_in_unit(unit.kelvin))
        print('#"%s"' % ('"\t"').join(headers), file=self._out)

    def __del__(self):
        if self._openedFile:
            self._out.close()

    @property
    def weights(self):
        return [x - self._weights[0] for x in self._weights]

    def step(self, steps):
        """Advance the simulation by integrating a specified number of time steps."""
        self.simulation.step(steps)

    def _attemptTemperatureChange(self, state):
        """Attempt to move to a different temperature."""

        # Compute the probability for each temperature.
        pot_ener = state.getPotentialEnergy()

        # Compute probability using:
        # p(n->m) = exp( (Bn-Bm)*E + fm-fn )
        temp_list = []
        prob_list = []
        min_i = 0
        index = self.currentTemperature

        # p(n->n-1)
        if self.currentTemperature != 0:
            min_i = index
            new_index = index - 1

            test_down = pot_ener * (
                self.inverseTemperatures[index] - self.inverseTemperatures[new_index]
            )
            test_down += self._weights[new_index] - self._weights[index]
            test_down = math.exp(test_down)

            temp_list.append(new_index)
            prob_list.append(test_down)

        # p(n->n+1)
        if self.currentTemperature < (len(self._weights) - 1):
            new_index = index + 1

            test_up = pot_ener * (
                self.inverseTemperatures[index] - self.inverseTemperatures[new_index]
            )
            test_up += self._weights[new_index] - self._weights[index]
            test_up = math.exp(test_up)

            # Make sure that highest p() is tested first
            if temp_list and (prob_list[0] < test_up):
                temp_list.insert(0, new_index)
                prob_list.insert(0, test_up)
            else:
                temp_list.append(new_index)
                prob_list.append(test_up)

        for temp_i, prob in zip(temp_list, prob_list):
            r = random.random()
            if r < prob:
                # print(f"SWITCH {self.currentTemperature:2} -> {temp_i:2}")
                # Rescale the velocities.
                scale = math.sqrt(
                    self.temperatures[temp_i]
                    / self.temperatures[self.currentTemperature]
                )
                velocities = scale * state.getVelocities(asNumpy=True).value_in_unit(
                    unit.nanometers / unit.picoseconds
                )
                self.simulation.context.setVelocities(velocities)

                # Select this temperature.
                self.currentTemperature = temp_i
                self.simulation.integrator.setTemperature(self.temperatures[temp_i])
                break

        if self._updateWeights:
            for k in range(min_i, len(self._weights) - 1):
                # Use Park and Pande weights:
                # f(n+1) = f(n) + (β(n+1) − β(n)) * (E(n+1) + E(n)) / 2,

                new_weight = (
                    self.inverseTemperatures[k + 1] - self.inverseTemperatures[k]
                )
                if self._e_pot_num[k + 1] != 0:
                    new_weight *= self._e_pot_sum[k] / (
                        2 * self._e_pot_num[k]
                    ) + self._e_pot_sum[k + 1] / (2 * self._e_pot_num[k + 1])
                    self._weights[k + 1] = (
                        self._weights[k] + new_weight * self._weights_factor[k + 1]
                    )

                # Use H. Nguyen hack
                # f(n+1) = f(n) + (β(n+1) − β(n)) * E(n),
                else:
                    new_weight *= self._e_pot_sum[k] / (self._e_pot_num[k])
                    self._weights[k + 1] = (
                        self._weights[k] + new_weight * self._weights_factor[k + 1]
                    )
                    break

        return

    def _writeReport(self):
        """Write out a line to the report."""
        temperature = self.temperatures[self.currentTemperature].value_in_unit(
            unit.kelvin
        )
        values = [temperature] + self.weights
        print(
            ("%d\t" % self.simulation.currentStep)
            + "\t".join("%g" % v for v in values),
            file=self._out,
        )
