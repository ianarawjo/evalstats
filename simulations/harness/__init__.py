"""Centralized simulation harness for evalstats.

This package consolidates the calibration/comparison Monte Carlo simulations
in ``simulations/*.py`` into a single CLI entry point (``python -m
simulations.harness.cli``) built on a shared synthetic-scenario library
(``scenarios/synthetic.py``) and a shared real-data adapter
(``scenarios/real_data.py``), so distributions and data sources can be
described once for the paper's Supplementary Material instead of once per
script. See ``simulations/harness/README.md`` for the case list and known
exceptions. The legacy scripts in ``simulations/`` are left untouched.
"""
