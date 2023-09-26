
[comment]: # (PyPSA README)

<!--
SPDX-FileCopyrightText: 2017-2023 The PyPSA-Eur Authors
SPDX-License-Identifier: CC-BY-4.0
-->

<!--![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/pypsa/pypsa-eur?include_prereleases)
-->
[![Build Status](https://github.com/pypsa/pypsa-eur/actions/workflows/ci.yaml/badge.svg)](https://github.com/PyPSA/pypsa-eur/actions)
<!--[![Documentation](https://readthedocs.org/projects/pypsa-eur/badge/?version=latest)](https://pypsa-eur.readthedocs.io/en/latest/?badge=latest)-->
![Size](https://img.shields.io/github/repo-size/centrefornetzero/PyPSA-eur)
[![Zenodo PyPSA-Eur](https://zenodo.org/badge/DOI/10.5281/zenodo.3520874.svg)](https://doi.org/10.5281/zenodo.3520874)
<!--[![Zenodo PyPSA-Eur-Sec](https://zenodo.org/badge/DOI/10.5281/zenodo.3938042.svg)](https://doi.org/10.5281/zenodo.3938042)-->
[![Snakemake](https://img.shields.io/badge/snakemake-≥5.0.0-brightgreen.svg?style=flat)](https://snakemake.readthedocs.io)
<!--[![REUSE status](https://api.reuse.software/badge/github.com/pypsa/pypsa-eur)](https://api.reuse.software/info/github.com/pypsa/pypsa-eur)-->
<!--[![Stack Exchange questions](https://img.shields.io/stackexchange/stackoverflow/t/pypsa)](https://stackoverflow.com/questions/tagged/pypsa).-->

# PyPSA-FES: A linear optimisation model to simulate Great Britain's energy transition

PyPSA-FES offers a complete data pipeline and optimisation backbone to model any year between 2023 and 2050 of Great Britain's energy transition both for an optimisic transition scenario _Leading the Way_, and a pessimistic version _Falling Short_. For the chosen scenario and year, the model dynamically retrieves constraints on generation, transmission and storage capacities, demand and emission targets from [_national grid ESO's Future Energy Scenarios_](https://www.nationalgrideso.com/future-energy/future-energy-scenarios), and runs a full year of hourly investment and operational optimisation for a 16-zonal network.

![PyPSA-Eur Grid Model](doc/img/fes_capacity_expansion.png)

Additionally, the model includes three types of domestic demand flexibility, that can be switched on or off
- **Demand Flexibility Service** individual households can provide to the grid by shifting their demand in time
- **Smart Heat Pumps** storing heat provided by heat pumps in homes' thermal inertia
- **Smart Charging Electric Vehicles** and **Vehicle-to-Grid** according to centrally optimised schedules.

![Energy Flows Modelled in PyPSA-FES](doc/img/pypsa_fes_flowchart.PNG)

The model is an adaption of the [PyPSA-Eur](github.com/pypsa/pypsa-eur), a sector-coupled european energy model, developed at [TU Berlin](https://www.tu.berlin/en/ensys), and is built on the underlying Python modelling library [PyPSA](github.com/pypsa/pypsa).


In this repo we explore whole-system implications of demand flexibility events as in [The Big Dirty Turn Down](https://www.centrefornetzero.org/work/energy-flexibility-domestic-research-octopus-energy/).

For this purpose, we make modifications to the energy model [PyPSA-Eur](github.com/pypsa/pypsa-eur), which is a linear model of the European energy system that optimises both operation and capacity expansion.

The present repo is a work in progress, but will feature interactive scenario presentations, and a ready-to-use pipeline to conduct various experiments.

For more details on the underlying PyPSA-Eur model, we refer to the [model documentation](https://pypsa-eur.readthedocs.io) or related research for instance [PyPSA-Eur: An Open Optimisation Model of the European Transmission System](https://arxiv.org/abs/1806.01613), 2018, [arXiv:1806.01613](https://arxiv.org/abs/1806.01613). The model building routines are defined through a snakemake workflow.  Please see the [documentation](https://pypsa-eur.readthedocs.io/) for installation instructions and other useful information about the snakemake workflow. The model is designed to be imported into the open toolbox [PyPSA](https://github.com/PyPSA/PyPSA).


The network is built up on the transmission network nodes from [PyPSA-Eur](https://github.com/PyPSA/pypsa-eur):

![network diagram](https://github.com/PyPSA/pypsa-eur/blob/master/doc/img/base.png?raw=true)

For computational reasons the model is usually clustered down
to 50-200 nodes.

Already-built versions of the model can be found in the accompanying [Zenodo
repository](https://doi.org/10.5281/zenodo.3601881).

# Contributing and Support
We strongly welcome anyone interested in contributing to this project. If you have any ideas, suggestions or encounter problems, feel invited to file issues or make pull requests on GitHub.
-   In case of code-related **questions**, please post on [stack overflow](https://stackoverflow.com/questions/tagged/pypsa).
-   For non-programming related and more general questions please refer to the [mailing list](https://groups.google.com/group/pypsa).
-   To **discuss** with other PyPSA users, organise projects, share news, and get in touch with the community you can use the [discord server](https://discord.com/invite/AnuJBk23FU).
-   For **bugs and feature requests**, please use the [PyPSA-Eur Github Issues page](https://github.com/PyPSA/pypsa-eur/issues).

# Licence

The code in this repository is released as free software under the [MIT License](https://opensource.org/licenses/MIT), see `LICENSE.txt`.  However, different licenses and terms of use may apply to the various input data.

[comment]: # (CentreForNetZero Python template)

# Python-Template

Centre for Net Zero's template for Python projects.

Tools:

* [Pipenv](https://github.com/pypa/pipenv) for dependency management
* [Pytest](https://github.com/pytest-dev/pytest/) for testing
* [Mypy](https://mypy.readthedocs.io/en/stable/) for type checking
* [Flake8](https://flake8.pycqa.org/en/latest/) for linting
* [isort](https://github.com/PyCQA/isort) and [black](https://github.com/psf/black) for formatting

There are two Github Actions workflows:

* `test_and_lint.yaml` runs checks on a Ubuntu Github-hosted runner.
* `container.yaml` runs the same checks but inside a Docker container and pushes images to [Google Cloud Platform Artifact Registry](https://cloud.google.com/artifact-registry).

## Secrets configuration

If you want to use the Docker workflow, you need to set the following secrets:

* `GCP_ARTIFACT_REGISTRY`, e.g. `LOCATION-docker.pkg.dev/PROJECT-ID`
* `GCP_ARTIFACT_REGISTRY_SA_KEY`, the key for a service account with the roles to push and pull images.

## Python Setup

You need to [set up your Python environment](https://docs.google.com/document/d/1Tg0eKalqOp-IJEeH7aShc9fYF5zn95H6jxEk25BLLUE/) first.

1. Clone this repo.
2. Run `make setup` in your terminal.

In step 2 this will:

* Run `pipenv sync --dev` to install dependencies.
* `cp .env.template .env` to create your local `.env` file - make sure you fill in any blanks.
* Install your local pre-commit environment which will be used to maintain code standards
* Check the tests work by running `pipenv run pytest`