Implementation
==============

The ST and SST2 protocol has been implemented by a Python
script using OpenMM(Eastman *et al.* 2017).
ST's original OpenMM script, written by Peter Eastman, was
modified to implement the weight calculation of
Park and Pande
and the *on the fly*
weight calculation from
Nguyen *et al.* 2013.
The same script
was also used to write the SST2 scripts.

The scaling of non-bonded interactions was done by scaling the
:math:`\epsilon` parameters of the Lennard-Jones potential 
by :math:`\lambda_m = \frac{\beta_m}{\beta_{ref}}`, while the charges of solute
atoms were scaled by :math:`\sqrt{\lambda_{m}}`. The 
solute intramolecular energies (bonds, angles, and improper torsions)
were left unchanged, for selected proper torsion terms,
the dihedral constant term :math:`k` was scaled by :math:`\lambda_m`.

In order to accurately compute the different energy terms :math:`E_{pp}`, :math:`E_{pw}`
and :math:`E_{ww}`, two additional molecular
systems were created in addition to the simulated systems,
consisting of the solute atoms only, and the
solvent atoms only. At each step where the energy terms had to be
computed, the additional systems were assigned the same coordinates as
the simulated systems. This allows the long
term electrostatic contribution of :math:`E_{pp}` and :math:`E_{ww}` to be accurately computed
and the the non-bonded interaction contribution :math:`E_{pw}` to be derived.