# Notes


## To improve/fix:

To Do:
- Try to remove entirely the `pdb_manip` call (creation of linear peptide)
- Clean documentation
- Add tests for tools
- Need to treat the case where weights is not None and restart_files is not None

BUG:
    - the xml files are overwritten when nsteps is reset to 0 (because of dcd format)
    - the restart nsteps is wrong when nsteps is reset to 0 (because of dcd format)

## Paper

To Do:
- check ST energie convergence after temperature change, with different coupling parameters
- Tref 350 K for SST2 : Running
- ST with same rung number as SST2
- SST2 with minimal rung number (~4)