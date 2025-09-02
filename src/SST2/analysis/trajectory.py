#!/usr/bin/env python3
# coding: utf-8

import os
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import MDAnalysis as mda
from MDAnalysis.analysis import align
from MDAnalysis.analysis import contacts
import MDAnalysis.analysis.pca as pca

from ipywidgets import IntProgress

# Logging
logger = logging.getLogger(__name__)


def read_traj(start_pdb, generic_name):

    traj_file_list = [f"{generic_name}.dcd"]

    # Get part number
    part = 1
    while os.path.isfile(f"{generic_name}_part_{part + 1}.csv"):
        part += 1
    logger.info(f" part number = {part}")

    for i in range(2, part + 1):
        traj_file_list.append(f"{generic_name}_part_{i}.dcd")

    md = mda.Universe(start_pdb, traj_file_list)

    return md


def prepare_traj(md, ref_Sel, compound="fragments"):

    prot = md.select_atoms(ref_Sel)
    rest = md.select_atoms(f"not ({ref_Sel})")

    print()
    f = IntProgress(min=0, max=md.trajectory.n_frames)  # instantiate the bar
    display(f)  # display the bar

    for ts in md.trajectory:
        # print(ts.frame)
        # all_sel.unwrap()
        protein_center = prot.center_of_geometry(pbc=False)
        dim = ts.triclinic_dimensions

        box_center = np.sum(dim, axis=0) / 2
        # print(f'Box: {box_center}  Prot: {protein_center}')
        md.atoms.translate(box_center - protein_center)
        rest.wrap(compound=compound)

        if ts.frame % 100 == 0:
            f.value += 100


def align_traj(md, ref, ref_Sel, tol_mass=0.1):

    alignment = align.AlignTraj(
        md, ref, select=f"{ref_Sel}", in_memory=True, verbose=True, tol_mass=tol_mass
    )
    _ = alignment.run()


def compute_native_contact(md, ref, sel="protein and not name H*", sel_2=None):

    ref.trajectory[-1]
    ref_atom_sel = ref.select_atoms(sel)

    if sel_2 is None:
        sel_2 = sel
        ref_atom_sel_2 = ref_atom_sel
    else:
        ref_atom_sel_2 = ref.select_atoms(sel_2)

    ca = contacts.Contacts(
        md,
        select=(sel, sel_2),
        refgroup=(ref_atom_sel, ref_atom_sel_2),
        kwargs={"beta": 5.0, "lambda_constant": 1.8},
        method="soft_cut",
    ).run(verbose=True)
    return ca.timeseries[:, 1]


def compute_pca(md, ref, sel="backbone", cum_var=0.8):

    prot_pca = pca.PCA(md, select=sel)
    logger.info("Compute PCA")
    prot_pca.run(verbose=True)

    # Get components defining cum_var % of variance
    n_pcs = max(2, np.where(prot_pca.cumulated_variance > cum_var)[0][0])

    logger.info(prot_pca.cumulated_variance)
    variance_pd = pd.DataFrame(prot_pca.cumulated_variance, columns=["variance"])
    variance_pd["PC"] = range(1, len(prot_pca.cumulated_variance) + 1)
    variance_pd["PC"] = variance_pd["PC"]

    plt.figure()
    ax = sns.barplot(data=variance_pd, x="PC", y="variance")
    plt.xlim(-0.5, 20)
    plt.ylim(0, 1.0)

    atomgroup = md.select_atoms(sel)

    pca_space = prot_pca.transform(atomgroup, n_components=n_pcs)

    col_names = ["PC_{}".format(i + 1) for i in range(n_pcs)]
    pca_df = pd.DataFrame(pca_space, columns=col_names)

    if ref is not None:
        logger.info("Compute PCA for ref structure")
        ref_atomgroup = ref.select_atoms(sel)
        pca_space_ref = prot_pca.transform(ref_atomgroup, n_components=n_pcs)

        pca_ref_df = pd.DataFrame(pca_space_ref, columns=col_names)
    else:
        pca_ref_df = None

    return (prot_pca, pca_df, pca_ref_df)
