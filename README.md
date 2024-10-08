[![Documentation Status](https://readthedocs.org/projects/sst2/badge/?version=latest)](https://sst2.readthedocs.io/en/latest/?badge=latest)
[![DOI:10.1101/2024.10.03.613476](http://img.shields.io/badge/DOI-10.1101/2024.10.03.613476-B31B1B.svg)](https://doi.org/10.1101/2024.10.03.613476)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13772542.svg)](https://doi.org/10.5281/zenodo.13772542)

<img src="https://raw.githubusercontent.com/samuelmurail/SST2/master/docs/source/logo.jpeg" alt="AF2 Analysis Logo" width="400" style="display: block; margin: auto;"/>

# Simulated Solute Tempering 2 (SST2)

## Description

This repository contains the source code for the Simulated Solute Tempering 2 (SST2) algorithm, as described in our paper ["Simulated Solute Tempering 2."](https://www.biorxiv.org/content/10.1101/2024.10.03.613476v1) SST2 is a novel enhanced sampling method for molecular dynamics (MD) simulations that combines the strengths of Simulated Tempering (ST) and Replica Exchange with Solute Tempering 2 (REST2).


* Source code repository on [gihub](https://github.com/samuelmurail/SST2)
* Documentation on readthedocs [![Documentation Status](https://readthedocs.org/projects/sst2/badge/?version=latest)](https://sst2.readthedocs.io/en/latest/?badge=latest)
* Manuscript on bioRxiv [![DOI:10.1101/2024.10.03.613476](http://img.shields.io/badge/DOI-10.1101/2024.10.03.613476-B31B1B.svg)](https://doi.org/10.1101/2024.10.03.613476)
* Trajectories on Zenodo [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13772542.svg)](https://doi.org/10.5281/zenodo.13772542)




## Algorithm Overview

SST2 aims to overcome limitations in conformational space exploration often encountered in traditional MD simulations, especially for systems with high energy barriers. SST2 builds on previous methods like REST2 and ST:

- **REST2:** Like REST2, SST2 scales solute-solute and solute-solvent interactions to enhance sampling.
- **ST:**  SST2 adopts the concept of single simulation traveling across different temperatures (or scaling factors) from ST, but applies it specifically to solute tempering as in REST2.

These features enable SST2 to effectively sample conformational space and provide valuable insights into the thermodynamics and kinetics of biomolecular systems.


## library Main features:

- ST simulation using Park and Pande weights calculation, and Phong et al. on the flight weights calculation.
- REST2 potential energy implementation
- **Efficient SST2 Sampling:** By selectively scaling solute interactions and exchanging replicas, SST2 significantly enhances sampling efficiency compared to traditional MD simulations.
- Binary scripts to run ST and SST2 simulations starting from an amino acid sequence or a `pdb` file.
- **Flexibility:** The algorithm allows users to adjust various parameters, including the number of rungs, the range of scaling factors, and the exchange frequency, to optimize performance for different systems.
- **Open Source:** The code is made available as open source, facilitating further development, adaptation, and application by the research community.

## Implementation

This implementation of SST2 utilizes the [OpenMM](https://openmm.org/) molecular dynamics library. The code is written in Python and utilizes OpenMM's custom forces and integrators to achieve the desired functionality.

## Contributing

`SST2` is an open-source project and contributions are welcome. If
you find a bug or have a feature request, please open an issue on the GitHub
repository at [https://github.com/samuelmurail/SST2](https://github.com/samuelmurail/SST2). If you would like
to contribute code, please fork the repository and submit a pull request.

## Author

* [Samuel Murail](https://samuelmurail.github.io/PersonalPage/), Associate Professor - [Université Paris Cité](https://u-paris.fr), [CMPLI](http://bfa.univ-paris-diderot.fr/equipe-8/).

See also the list of [contributors](https://github.com/samuelmurail/SST2/contributors) who participated in this project.

## License

This project is licensed under the GNU General Public License v2.0 - see the ``LICENSE`` file for details.

## Citation

If you use this code in your research, please cite our paper:

- [Stratmann D, Moroy G, Tuffery P and Murail S. Simulated Solute Tempering 2.
*bioRxiv* 2024, doi:10.1101/2024.10.03.613476.](https://www.biorxiv.org/content/10.1101/2024.10.03.613476v1)
