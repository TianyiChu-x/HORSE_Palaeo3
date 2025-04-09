# HORSE_Palaeo3
HORizon SEquencing


# HORSE Tests

This repository contains various scripts for testing and comparing computation results from HORSE and/or CONOP computational methods.

## Scripts

### sequence_comparison.py
Used to compare output sequences between two computations (HORSE and/or CONOP):
- Calculates correlation coefficients between sequences
- Generates range charts for visual comparison
- Helps in analyzing differences between computational methods

### mult_runs.py
Evaluates the robustness of HORSE computation results:
- Runs multiple HORSE computations on the same dataset
- Analyzes consistency across different runs
- Provides statistical measures of result stability

### iteration_track.py
Visualizes the progression of HORSE computations:
- Tracks changes in penalty across iterations
- Monitors correlation to ground truth throughout the computation process
- Creates graphs showing convergence patterns

### pseudo_ha.py
Generates artificial datasets for computational testing:
- Creates synthetic data for HORSE and HA algorithm evaluation

### utils.py
Contains utility functions used by the other scripts.