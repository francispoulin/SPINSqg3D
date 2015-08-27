# SPINSqg3D
A SPINS script that solves the stratified 3D Quasi Geostrophic model

The files in this repository are:

1. qg3D.cpp
  * Case file using SPINS to perform 3D QG simulations. Compile using the normal SPINS procedure.
2. config_qg3d.py
  * Configuration tool for running simulations. Produces data files for initial conditions and a spins.conf file.
  * The majority of simulation parameters are specified in the config_qg3d.py script and so recompiling the source code is often not required.
3. movies_qg.py
  * Creates animations of the full and perturbation potential vorticity at mid-depth and y=Ly/2
  * Note: parallelized 
4. Compute_norms.py
  * Plots the norm of the perturbation potential vorticity as well as the growth rate.
  * Note: parallelized
