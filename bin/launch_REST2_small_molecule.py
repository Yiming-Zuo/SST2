#!/usr/bin/env python3

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np

import openmm
from openmm import app, unit

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from SST2.rest2 import REST2, run_rest2  # noqa: E402
import SST2.tools as tools  # noqa: E402


logger = logging.getLogger(__name__)


DEFAULT_SOLVENT_RESNAMES = {
    "HOH",
    "H2O",
    "OH2",
    "WAT",
    "SOL",
    "TIP3",
    "TIP3P",
    "TIP3PF",
    "TIP4",
    "TIP4P",
    "TIP4PEW",
    "TIP4PFB",
    "TIP5P",
    "TIP5PEW",
    "SPC",
    "SPCE",
    "OPC",
    "OPC3",
    "WT4",
}

DEFAULT_ION_RESNAMES = {
    "NA",
    "NA+",
    "CL",
    "CL-",
    "K",
    "K+",
    "CA",
    "CA2+",
    "MG",
    "MG2+",
    "ZN",
    "ZN2+",
    "POT",
    "CLA",
    "SOD",
    "CES",
    "RB",
    "SR",
}


def detect_best_platform():
    """自动检测最佳可用的 OpenMM 平台

    优先级：CUDA > OpenCL > CPU

    Returns
    -------
    str
        检测到的最佳平台名称
    """
    for platform_name in ["CUDA", "OpenCL", "CPU"]:
        try:
            platform = openmm.Platform.getPlatformByName(platform_name)
            logger.info(f"Detected available platform: {platform_name}")
            return platform_name
        except Exception:
            logger.debug(f"Platform {platform_name} not available")
            continue

    logger.warning("No GPU platform detected, falling back to CPU")
    return "CPU"


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run a REST2 simulation for a solvated small molecule."
    )
    parser.add_argument(
        "-pdb",
        dest="structure",
        required=True,
        help="Input PDB or mmCIF file containing the solute (no solvent).",
    )
    parser.add_argument(
        "-n",
        "--name",
        dest="name",
        required=True,
        help="Base name for generated files.",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        dest="out_dir",
        required=True,
        help="Directory where intermediate and output files will be written.",
    )
    parser.add_argument(
        "--ff",
        default="amber14sb",
        help="Protein force-field family used to build the OpenMM ForceField (default: %(default)s).",
    )
    parser.add_argument(
        "--water-ff",
        dest="water_ff",
        default="tip3p",
        help="Water model to pair with --ff (default: %(default)s).",
    )
    parser.add_argument(
        "--extra-ff",
        dest="extra_ff",
        nargs="+",
        default=None,
        help="Additional XML force-field files (e.g. ligand parameters).",
    )
    parser.add_argument(
        "--solute-residues",
        nargs="+",
        default=None,
        help="Residue names to treat as the solute. "
        "Defaults to all residues that are not recognised as solvent or ions.",
    )
    parser.add_argument(
        "--extra-solvent-residues",
        nargs="+",
        default=None,
        help="Additional residue names to treat as solvent/ions.",
    )
    parser.add_argument(
        "--pad",
        type=float,
        default=1.5,
        help="Solvent padding (nm) when creating the water box (default: %(default)s nm).",
    )
    parser.add_argument(
        "--ionic-strength",
        type=float,
        default=0.15,
        help="Ionic strength (M) for the solvated system (default: %(default)s M).",
    )
    parser.add_argument(
        "--positive-ion",
        default="Na+",
        help="Positive ion used for neutralisation (default: %(default)s).",
    )
    parser.add_argument(
        "--negative-ion",
        default="Cl-",
        help="Negative ion used for neutralisation (default: %(default)s).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=300.0,
        help="Target temperature in Kelvin (default: %(default)s K).",
    )
    parser.add_argument(
        "--pressure",
        type=float,
        default=1.0,
        help="Pressure in atmospheres (default: %(default)s atm).",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=10.0,
        help="Production time in nanoseconds for REST2 sampling (default: %(default)s ns).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=2.0,
        help="Integration timestep in femtoseconds. With constraints=HBonds, "
        "2 fs is safe for most small molecules without HMR (default: %(default)s fs).",
    )
    parser.add_argument(
        "--friction",
        type=float,
        default=1.0,
        help="Langevin friction coefficient in ps^-1 (default: %(default)s ps^-1).",
    )
    parser.add_argument(
        "--hmr",
        type=float,
        default=1.0,
        help="Hydrogen mass repartitioning (amu) applied during system construction. "
        "Use 1.0 to disable (recommended for small molecules), 1.2-1.5 for speed "
        "(may affect vibrational frequencies and dynamics) (default: %(default)s amu).",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=1.0,
        help="Non-bonded cutoff distance (nm) (default: %(default)s nm).",
    )
    parser.add_argument(
        "--nonbonded-method",
        dest="nonbonded_method",
        choices=["pme", "cutoff-periodic"],
        default="pme",
        help="Method for long-range electrostatics. Use 'cutoff-periodic' (Reaction Field) for charged solutes (default: %(default)s).",
    )
    parser.add_argument(
        "--platform",
        default=None,
        help="OpenMM platform (CUDA/OpenCL/CPU). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--dcd-interval",
        type=float,
        default=0.5,
        help="Trajectory save interval in picoseconds (default: %(default)s ps).",
    )
    parser.add_argument(
        "--log-interval",
        type=float,
        default=0.5,
        help="State data log interval in picoseconds (default: %(default)s ps).",
    )
    parser.add_argument(
        "--rest2-interval",
        type=float,
        default=0.5,
        help="REST2 energy report interval in picoseconds (default: %(default)s ps).",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=float,
        default=None,
        help="Checkpoint interval in picoseconds. Disabled if not provided.",
    )
    parser.add_argument(
        "--minimize",
        action="store_true",
        help="Perform an energy minimisation before production.",
    )
    parser.add_argument(
        "--minimize-iterations",
        type=int,
        default=5000,
        help="Maximum iterations for the optional minimisation (default: %(default)s).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite intermediate files if they already exist.",
    )
    parser.add_argument(
        "--use-fixer",
        action="store_true",
        help="Use pdbfixer preprocessing (useful for peptides/proteins). Off by default to preserve small-molecule templates.",
    )
    parser.add_argument(
        "--ph",
        type=float,
        default=7.0,
        help="pH used by pdbfixer when --use-fixer is enabled (default: %(default)s).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    return parser


def copy_input(structure_path: Path, destination: Path, overwrite: bool) -> Path:
    if destination.exists() and not overwrite:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(structure_path, destination)
    return destination


def determine_solute_indices(topology, explicit_residues, solvent_residues):
    explicit = {res.upper() for res in explicit_residues} if explicit_residues else None
    solvent = {res.upper() for res in solvent_residues}

    solute_indices = []
    for atom in topology.atoms():
        residue_name = atom.residue.name.strip().upper()
        if explicit:
            if residue_name in explicit:
                solute_indices.append(int(atom.index))
        else:
            if residue_name not in solvent:
                solute_indices.append(int(atom.index))

    return solute_indices


def interval_to_steps(interval_ps, timestep_fs, label="Interval"):
    """将时间间隔转换为步数，并检查舍入误差

    Parameters
    ----------
    interval_ps : float
        时间间隔（皮秒）
    timestep_fs : float
        时间步长（飞秒）
    label : str
        间隔标签，用于日志输出

    Returns
    -------
    int
        对应的步数（至少为1）
    """
    exact = interval_ps * 1000.0 / timestep_fs
    steps = max(1, int(np.round(exact)))

    # 检查舍入误差
    if abs(exact - steps) > 0.01:
        actual_time = steps * timestep_fs / 1000.0
        logger.warning(
            f"{label}: requested {interval_ps:.3f} ps, "
            f"but using {steps} steps = {actual_time:.3f} ps "
            f"(rounded due to timestep mismatch)"
        )

    return steps


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    logger.info("Starting REST2 small-molecule workflow.")

    input_path = Path(args.structure).expanduser().resolve()
    if not input_path.exists():
        parser.error(f"Input structure {input_path} does not exist.")

    if input_path.suffix.lower() not in {".pdb", ".cif", ".mmcif"}:
        parser.error("Input structure must be a PDB or mmCIF file.")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    name = args.name
    base_path = out_dir / name

    if args.use_fixer:
        fixed_cif = out_dir / f"{name}_fixed.cif"
        tools.prepare_pdb(
            str(input_path), str(fixed_cif), pH=args.ph, overwrite=args.overwrite
        )
        prepared_input = fixed_cif
    else:
        prepared_input = out_dir / f"{name}_prepared{input_path.suffix.lower()}"
        copy_input(input_path, prepared_input, args.overwrite)

    logger.info("Loading force-field definitions.")
    forcefield = tools.get_forcefield(args.ff, args.water_ff, extra_files=args.extra_ff)

    solvated_cif_path = out_dir / f"{name}_solvated.cif"
    logger.info("Building solvated system with REST2-compatible topology.")
    ionic_strength = args.ionic_strength * unit.molar
    solvated_cif = tools.create_water_box(
        str(prepared_input),
        str(solvated_cif_path),
        forcefield=forcefield,
        pad=args.pad * unit.nanometer,
        ionicStrength=ionic_strength,
        positiveIon=args.positive_ion,
        negativeIon=args.negative_ion,
        overwrite=args.overwrite,
    )

    solvated_pdb_path = out_dir / f"{name}_solvated.pdb"
    with open(solvated_pdb_path, "w") as handle:
        app.PDBFile.writeFile(
            solvated_cif.topology,
            solvated_cif.positions,
            handle,
            keepIds=True,
        )

    solvent_residues = set(DEFAULT_SOLVENT_RESNAMES | DEFAULT_ION_RESNAMES)
    if args.extra_solvent_residues:
        solvent_residues.update(res.upper() for res in args.extra_solvent_residues)

    solute_indices = determine_solute_indices(
        solvated_cif.topology, args.solute_residues, solvent_residues
    )

    if not solute_indices:
        parser.error(
            "No solute atoms were detected. "
            "Provide --solute-residues or adjust --extra-solvent-residues."
        )

    solute_index_set = set(solute_indices)
    solute_residues = {
        atom.residue.name.strip()
        for atom in solvated_cif.topology.atoms()
        if int(atom.index) in solute_index_set
    }

    logger.info(
        "Identified %d solute atoms across %d residues.",
        len(solute_indices),
        len(solute_residues),
    )

    temperature = args.temperature * unit.kelvin
    pressure = args.pressure * unit.atmospheres
    dt_fs = args.dt
    dt = dt_fs * unit.femtoseconds
    friction = args.friction / unit.picoseconds
    hydrogen_mass = args.hmr * unit.amu
    cutoff = args.cutoff * unit.nanometer

    integrator = openmm.LangevinMiddleIntegrator(temperature, friction, dt)

    # 设置 nonbonded method（在创建 system 之前）
    if args.nonbonded_method == "pme":
        nonbonded_method = app.PME
        logger.info("Using PME for long-range electrostatics")
    else:
        nonbonded_method = app.CutoffPeriodic
        logger.info("Using CutoffPeriodic (Reaction Field) for long-range electrostatics")

    logger.info("Creating OpenMM system object...")
    try:
        system = tools.create_sim_system(
            solvated_cif,
            forcefield=forcefield,
            temp=temperature,
            h_mass=hydrogen_mass,
            rigidWater=True,
            constraints=app.HBonds,
            nonbondedMethod=nonbonded_method,
            nonbondedCutoff=cutoff,
        )
    except Exception as e:
        logger.error(
            "Failed to create OpenMM system. Common causes:\n"
            "  - Missing force-field parameters for solute residues\n"
            "  - Incompatible residue/atom names with force-field\n"
            "  - Unrecognised chemical groups in the structure\n"
            f"\nOriginal error: {type(e).__name__}: {e}"
        )
        sys.exit(1)

    # 检查溶质电荷
    logger.info("Checking solute charge...")
    nonbonded = None
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            nonbonded = force
            break

    if nonbonded:
        charges = []
        for i in range(system.getNumParticles()):
            charge, sigma, epsilon = nonbonded.getParticleParameters(i)
            charges.append(charge.value_in_unit(unit.elementary_charge))

        total_charge = sum(charges)
        solute_charge = sum(charges[i] for i in solute_indices)

        logger.info(f"System total charge: {total_charge:.2f} e")
        logger.info(f"Solute charge: {solute_charge:.2f} e")

        if abs(total_charge) > 0.01:
            logger.error(
                f"System is not neutral (total charge = {total_charge:.2f} e). "
                "Add counter-ions or check input structure."
            )
            sys.exit(1)

        if abs(solute_charge) > 0.01 and args.nonbonded_method == "pme":
            logger.warning(
                f"\n{'='*70}\n"
                f"⚠️  WARNING: Charged solute detected!\n"
                f"{'='*70}\n"
                f"Solute has net charge: {solute_charge:.2f} e\n"
                f"\n"
                f"PME with REST2 may introduce artifacts for charged solutes.\n"
                f"\n"
                f"Recommendations:\n"
                f"  1. Use --nonbonded-method cutoff-periodic (Reaction Field)\n"
                f"  2. Ensure OpenMM version >= 8.3.1 if using PME\n"
                f"  3. Validate energy conservation in short test runs\n"
                f"{'='*70}\n"
            )

    # 设置平台
    if args.platform is None:
        platform_name = detect_best_platform()
        logger.info(f"Using auto-detected platform: {platform_name}")
    else:
        platform_name = args.platform
        logger.info(f"Using user-specified platform: {platform_name}")

    logger.info("Initialising REST2 machinery.")
    rest2_system = REST2(
        system=system,
        pdb=solvated_cif,
        forcefield=forcefield,
        solute_index=solute_indices,
        integrator=integrator,
        platform_name=platform_name,
        temperature=temperature,
        pressure=pressure,
        dt=dt,
        friction=friction,
        hydrogenMass=hydrogen_mass,
        nonbondedMethod=nonbonded_method,
        nonbondedCutoff=cutoff,
        constraints=app.HBonds,
        rigidWater=True,
        ewaldErrorTolerance=0.0005,
    )

    if args.minimize:
        logger.info(
            "Performing energy minimisation (max %d iterations).",
            args.minimize_iterations,
        )
        rest2_system.simulation.minimizeEnergy(maxIterations=args.minimize_iterations)

    total_steps = int(np.ceil(args.time * 1_000_000.0 / dt_fs))

    logger.info("Planning REST2 production for %d integration steps.", total_steps)

    dcd_interval = interval_to_steps(args.dcd_interval, dt_fs, "DCD save interval")
    log_interval = interval_to_steps(args.log_interval, dt_fs, "Log interval")
    rest2_interval = interval_to_steps(args.rest2_interval, dt_fs, "REST2 report interval")
    checkpoint_interval = (
        interval_to_steps(args.checkpoint_interval, dt_fs, "Checkpoint interval")
        if args.checkpoint_interval is not None
        else None
    )

    logger.info("Launching REST2 sampling.")
    run_rest2(
        rest2_system,
        str(base_path),
        total_steps,
        dt,
        save_step_dcd=dcd_interval,
        save_step_log=log_interval,
        save_step_rest2=rest2_interval,
        overwrite=args.overwrite,
        save_checkpoint_steps=checkpoint_interval,
    )

    logger.info("REST2 run completed. Outputs written under %s", base_path)


if __name__ == "__main__":
    main()
