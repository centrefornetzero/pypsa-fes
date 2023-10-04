..
  SPDX-FileCopyrightText: 2019-2023 The PyPSA-Eur Authors, Lukas Franken

  SPDX-License-Identifier: CC-BY-4.0

.. _installation:

##########################################
Installation
##########################################


Clone the Repository
====================

First of all, clone the `PyPSA-FES repository <https://github.com/centrefornetzero/pypsa-fes>`_
using the version control system ``git`` in the command line.

.. code:: bash

    /some/other/path % cd /some/path

    /some/path % git clone https://github.com/centrefornetzero/pypsa-fes.git


The model draws its assumptions about technologies from a separate repository, it needs to be next to 
the model directory ``pypsa-fes``.

.. code:: bash

    /some/path % git clone https://github.com/pypsa/technology-data.git


.. _deps:

Install Python Dependencies
===============================

PyPSA-Eur relies on a set of other Python packages to function.
We recommend using the package manager `mamba <https://mamba.readthedocs.io/en/latest/>`_ to install them and manage your environments.
For instructions for your operating system follow the ``mamba`` `installation guide <https://mamba.readthedocs.io/en/latest/installation.html>`_.
You can also use ``conda`` equivalently.

The package requirements are curated in the `envs/environment.yaml <https://github.com/PyPSA/pypsa-fes/blob/master/envs/environment.yaml>`_ file.
The environment can be installed and activated using

.. code:: bash

    .../pypsa-eur % mamba env create -f envs/environment.yaml

    .../pypsa-eur % mamba activate pypsa-eur

.. note::
    The equivalent commands for ``conda`` would be

    .. code:: bash

        .../pypsa-eur % conda env create -f envs/environment.yaml

        .../pypsa-eur % conda activate pypsa-eur

.. note::
    **For MAC OS users**, it appears using `micromamba <https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html>`_ instead of ``mamba``
    works better. In that case, we recommend installing the fixed environment using

    .. code:: bash

        .../pypsa-fes % micromamba env create -f envs/environment.fixed.yaml

        .../pypsa-fes % micromamba activate pypsa-eur


Install a Solver
================

PyPSA passes the PyPSA-FES network model to an external solver to perform the optimisation.

Generally, `Gurobi <https://www.gurobi.com/documentation/quickstart.html>`_ 
is recommended, however only academic licences are free.
Otherwise, the model supports the almost equally strong open-source solver `HiGHS <https://highs.dev/>`_.

To install the solvers, please refer to the respective installation guides and/or run

    For HiGHS, run

    .. code:: bash

        mamba activate pypsa-eur
        mamba install -c conda-forge ipopt
        pip install highspy

    For Gurobi, run

    .. code:: bash

        mamba activate pypsa-eur
        mamba install -c gurobi gurobi

    Additionally, you need to setup your `Gurobi license <https://www.gurobi.com/solutions/licensing/>`_.


.. _defaultconfig:

Handling Configuration Files
============================

PyPSA-FES has several configuration options that must be specified in a
``config/config.yaml`` file located in the root directory. An example configuration
``config/config.default.yaml`` is maintained in the repository, which will be used to
automatically create your customisable ``config/config.yaml`` on first use. More
details on the configuration options are in :ref:`config`.

You can also use ``snakemake`` to specify another file, e.g.
``config/config.mymodifications.yaml``, to update the settings of the ``config/config.yaml``.

.. code:: bash

    .../pypsa-fes % snakemake -call --configfile config/config.mymodifications.yaml

.. warning::
    Users are advised to regularly check their own ``config/config.yaml`` against changes
    in the ``config/config.default.yaml`` when pulling a new version from the remote
    repository.