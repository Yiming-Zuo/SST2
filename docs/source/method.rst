Method Description
==================

The Simulated Solvent Tempering 2 (SST2) algorithm is an enhanced sampling method for molecular dynamics (MD) simulations. It combines aspects of Simulated Tempering (ST)
and Replica Exchange with Solute Tempering 2 (REST2) to overcome limitations in traditional MD simulations when dealing with systems that have high energy barriers. The key aim of SST2 is to enhance the exploration of conformational space and gain a better understanding of the thermodynamics and kinetics of biomolecular systems.

We used a slightly modified version of the original implementation of
REST2. In the approach proposed 
in Stirnemann and Sterpone, only the proper torsion
terms of the solute were scaled among the bonded terms:

.. math::
    :label: SST2

    E_{m}^{SST2} (X) = \frac{\beta_m}{\beta_{ref}} E_{pp}^{(1)}(X)
    + E_{pp}^{(2)}(X) + \\
    \sqrt{\frac{\beta_m}{\beta_{ref}}} E_{pw}(X)
    +  E_{ww}(X)

or using $\lambda_m = \sfrac{\beta_m}{\beta_{ref}}$:

.. math::
    $E_{m}^{SST2} (X) = \lambda_m E_{pp}^{(1)}(X) + E_{pp}^{(2)}(X) + \\
    \sqrt{\lambda_m} E_{pw}(X) +  E_{ww}(X)


where :math:`E_{pp}^{(1)}` is the scaled solute intramolecular energy
(LJ, Coulomb, and proper torsions), :math:`E_{pp}^{(2)}` is the unscaled
solute intramolecular energy (bonds, angles, and improper torsions),
:math:`E_{pw}` is the solute-solvent interaction energy and :math:`E_{ww}` is the
solvent intramolecular energy.

The acceptance ratio is given by :math:`p_{mn} = \min (1, e^{\Delta_{mn}^{SST2}})`
, where inserting (:eq:`SST2`) into the ST formula :math:`\Delta_{mn}^{ST} =  (\beta_m - \beta_n) E + (w_n - w_m)`
gives:

.. math::
    \Delta_{mn}^{SST2} = (\beta_m-\beta_n) \bigg[E_{pp}^{(1)}(X) + \\
    \frac{\sqrt{\beta_{ref}}}{ \sqrt{\beta_m}  + \sqrt{\beta_n}} E_{pw}
    (X)\bigg] +(w_n- w_m)


This equation is equivalent to the original SST methods (Denschlag *et al.*).

Using :math:`\Delta_{mn}^{typ} = \Delta_{nm}^{typ}`, we obtain:

.. math::
    (w_n - w_m) = (\beta_n - \beta_m) \frac{ (\braket{E_{pp}^{(1)}}_m
    -  \braket{E_{pp}^{(1)}}_n)}{2} + \\
    (\sqrt{\beta_{ref} \beta_n} -
    \sqrt{\beta_{ref} \beta_m}) \frac {(\braket{E_{pw}}_m -
    \braket{E_{pw}}_n)}{2}


The exchange with neighboring :math:`m + 1` replicas is determined by
the fluctuation of :math:`E_{pp} + \frac{\sqrt{\beta_{ref}}} {\big(\sqrt{\beta_m} + \sqrt{\beta_{m+1}}\big)} E_{pw}` or for
simplicity, since :math:`\beta_m` and :math:`\beta_{m+1}` are close, we will later
monitor :math:`E_{pp} + 0.5 \sqrt{\frac{\beta_{ref}}{\beta_m}} E_{pw}`.

