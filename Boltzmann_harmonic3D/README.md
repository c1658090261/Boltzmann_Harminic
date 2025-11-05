# LBM harmonic User Guide

## Introduction  
LBM harmonic is a Single-Core 3D Harmonic Lattice Boltzmann solver.

## System Requirements  
- **OS**: Ubuntu 22.04 or newer  
- **Compiler**: GCC ≥ 12 (C++17 capable)  
- **Dependencies**: `gcc`

## Build & Run  
1. **Configure & compile**  
   ```shell
   g++ Boltzmann_harmonic3D.cpp
   ```

2. **Launch the simulation**  
   ```shell
   ./a.out
   ```

## Output Files  
- `3D-Laplace*.dat`: cross-sectional field data  
- `subtraction*.dat`: residual history  

Follow the steps above to experience the LBM harmonic solver.