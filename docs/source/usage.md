# Basic usage

## Scripts

We provide several python script to use `SST2` in a project, the script are located in the `bin` directory of the `SST2` package. The scripts are:

* `launch_ST_abinitio_seq.py`
* `launch_ST_pdb.py`
* `launch_sst2_abinitio_seq.py`
* `launch_sst2_pdb.py`

The other scripts are experimental and should be used with caution.

all scripts use:

* Amber99sbnmr force-field for implicit solvent simulations
* Amber14SB force-field for explicit solvent simulations, with TIP3P water model
* Use the `pdbfixer` to fix the pdb files and assign a **pH=7.0** protonation state to the peptide.

If you want to use another force-field or pH, you can modify the scripts accordingly.

## Available options

The scripts have several options that can be displayed by using the `--help` option. For example:

```bash
$ python bin/launch_ST_abinitio_seq.py --help
usage: launch_ST_abinitio_seq.py [-h] -seq SEQ -n NAME -dir OUT_DIR [-pad PAD]
                                 [-eq_time_impl EQ_TIME_IMPL] [-eq_time_expl\
...
...
```

## Folding ST simulations

* Here is an example to launch a ST simulation of a protein with a given sequence:

```bash
python bin/launch_ST_abinitio_seq.py  -seq NLYIQWLKDGGPSSGRPPPS\
    -time 1000 -temp_time 4 -min_temp 280 -last_temp 600\
    -n TrpCage -dir tmp_TrpCage
```

This command will perform a ST simulation of the TrpCage protein with the sequence `NLYIQWLKDGGPSSGRPPPS`. For *ab initio* simulations, an linear structure of the peptide is created and equilibrated in implicit solvent for 10ns, the system is then solvated and equilibrated in explicit solvent for 10 ns. A 1000 ns ST simulation will then be launched. 
ST will used temperatures distributed exponentially between 280 K to 600 K, with a temperature time change interval of 4 ps. The results will be saved in the `tmp_TrpCage` directory.

* Here is an example to launch a ST simulation of a protein from a given pdb:

```bash
python bin/launch_ST_pdb.py  -pdb my_structure.pdb -time 1000\
    -temp_time 4 -min_temp 280 -last_temp 600 -n TrpCage\
    -dir tmp_TrpCage
```

Here the implicit solvent equilibration is skipped, the system is directly solvated and equilibrated in explicit solvent for 10 ns. The rest of the simulation is the same as the previous example.

## Folding SST2 simulations

* To launch SST2 simulation of a protein with a given sequence:

```bash
python bin/launch_sst2_abinitio_seq.py  -seq NLYIQWLKDGGPSSGRPPPS\
 -time 1000 -temp_time 4 -min_temp 280 -ref_temp 320 -last_temp 600\
  -n TrpCage -dir tmp_SST2_TrpCage -exclude_Pro_omega
```

This command will perform a SST2 simulation of the TrpCage protein with the sequence `NLYIQWLKDGGPSSGRPPPS`. For *ab initio* simulations, an linear structure of the peptide is created and equilibrated in implicit solvent for 10ns, the system is then solvated and equilibrated in explicit solvent for 10 ns. A 1000 ns STT2 simulation will then be launched. 
STT2 will used temperatures distributed exponentially between 280 K to 600 K with a reference temperature of 320 K, the temperature time change interval is 4 ps. The results will be saved in the `tmp_SST2_TrpCage` directory.

Here we use the `-exclude_Pro_omega` option to exclude proline {math}`\omega` angles from SST2 scaling.

* SST2 simulation of a protein from a given pdb:

```bash
python bin/launch_sst2_pdb.py  -pdb my_structure.pdb -time 1000\
 -temp_time 4 -min_temp 280 -ref_temp 320 -last_temp 600 -n TrpCage\
  -dir tmp_SST2_TrpCage -exclude_Pro_omega
```

STT2 will used temperatures distributed exponentially between 280 K to 600 K with a reference temperature of 320 K, the temperature time change interval is 4 ps. The results will be saved in the `tmp_SST2_TrpCage` directory.

## Binding SST2 simulations

* To launch SST2 simulation of a protein-ligand complex with a given structure:

```bash
python bin/launch_sst2_pdb.py  -pdb my_structure.pdb -time 1000\
 -temp_time 4 -min_temp 280 -ref_temp 320 -last_temp 600 -n Complex\
  -dir tmp_SST2_Complex -chain B
```

- The `-chain B` option is used to specify the chain of the ligand or *solute* in the pdb file.
