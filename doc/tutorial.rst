..
  SPDX-FileCopyrightText: 2019-2023 The PyPSA-Eur Authors, Lukas Franken

  SPDX-License-Identifier: CC-BY-4.0

.. _tutorial:

###########################
Tutorial: Running the model
###########################

.. note::
    If you have not done it yet, follow the :ref:`installation` steps first.

Before getting started with **PyPSA-FES** it makes sense to be familiar
with its general modelling framework `PyPSA <https://pypsa.readthedocs.io>`__.

Generally, the model is suitable for laptops, even though at least 16 GB RAM
is recommended. The tutorial will cover examples on how to configure and
customise the PyPSA-FES model and run the ``snakemake`` workflow.

.. code:: bash
    :class: full-width

    snakemake -call results/octopus_fes/networks/elec_s_ec_lv1.5___LW_2030.nc --configfile config/config.default.yaml

This command will run the model for a single scenario and store
the results in ``results/octopus_fes/networks``.

Crucially, this command defines which scenario and year is run. Here, ``LW``
refers to *Leading the Way*, (``FS`` refers to *Falling Short*) and ``2030``
defines the modelled year.

..
    How to configure runs?
    ======================

    .. literalinclude:: ..config/config.default.yaml
    :language: yaml
    :start-at: countries:
    :end-before: snapshots:
