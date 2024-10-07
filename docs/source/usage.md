# Basic usage

We provide several python script to use `SST2` in a projectm, the script are located in the `bin` directory of the `SST2` package. The scripts are:

* `launch_ST_abinitio_seq.py`
* `launch_ST_pdb.py`
* `launch_sst2_abinitio_seq.py`
* `launch_sst2_pdb.py`

The other scripts are experimental and should be used with caution.

## Available options

The scripts have several options that can be displayed by using the `--help` option. For example:

```bash
$ python bin/launch_ST_abinitio_seq.py --help
usage: launch_ST_abinitio_seq.py [-h] -seq SEQ -n NAME -dir OUT_DIR [-pad PAD]
                                 [-eq_time_impl EQ_TIME_IMPL] [-eq_time_expl EQ_TIME_EXPL]
                                 [-time TIME] [-temp_list TEMP_LIST [TEMP_LIST ...]]
                                 [-temp_time TEMP_TIME] [-log_time LOG_TIME] [-min_temp MIN_TEMP]
                                 [-last_temp LAST_TEMP] [-hmr HMR] [-temp_num TEMP_NUM]
                                 [-friction FRICTION]

Simulate a peptide starting from a linear conformation.

options:
  -h, --help            show this help message and exit
  -seq SEQ              Input Sequence
  -n NAME               Output file name
  -dir OUT_DIR          Output directory for intermediate files
  -pad PAD              Box padding, default=1.5 nm
  -eq_time_impl EQ_TIME_IMPL
                        Implicit solvent Equilibration time, default=10 (ns)
  -eq_time_expl EQ_TIME_EXPL
                        Explicit Solvent Equilibration time, default=10 (ns)
  -time TIME            ST time, default=10.000 (ns)
  -temp_list TEMP_LIST [TEMP_LIST ...]
                        SST2 temperature list, default=None
  -temp_time TEMP_TIME  ST temperature time change interval, default=2.0 (ps)
  -log_time LOG_TIME    ST log save time interval, default= temp_time=2.0 (ps)
  -min_temp MIN_TEMP    Base temperature, default=300(K)
  -last_temp LAST_TEMP  Base temperature, default=500(K)
  -hmr HMR              Hydrogen mass repartition, default=3.0 a.m.u.
  -temp_num TEMP_NUM    Temperature rung number, default=None (computed as function of Epot)
  -friction FRICTION    Langevin Integrator friction coefficient default=10.0 (ps-1)
```

