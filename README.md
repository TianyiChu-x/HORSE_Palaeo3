# HORSE_Palaeo3
HORizon SEquencing

This repository contains two main components:
1. The HORSE C++ program for horizon sequencing
2. Python scripts for testing and comparing computation results from HORSE and/or CONOP methods

# How to Use HORSE

## System Requirements
- Currently only supports Mac with M series chips (M1, M2, etc.)

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/HORSE_Palaeo3.git
   cd HORSE_Palaeo3
   ```

2. Install OpenMP (required dependency):
   ```
   brew install openmp
   ```

3. Run HORSE with help flag to see available options:
   ```
   ./HORSE -h
   ```

## HORSE Options
- `-h, --help`: Show help message
- `-i <path>`: Input CSV file path
- `-g <int>`: Number of generations (default: 1000)
- `-p <int>`: Number of population (default: 10000)
- `-t <double>`: Teaser value (default: 0.01)
- `-s <int>`: Save rate (default: 1000)

# HORSE Tests

This repository contains various scripts for testing and comparing computation results.

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