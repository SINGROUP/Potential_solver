================
PyPointChargePotentialSolver
================
Created maanantai 19 joulukuu 2016

Description
-----------
A Python module for solving electrostatic potential of a distribution of point charges using periodic boundary conditions. Represents point charges as Gaussian charge distributions. The values of the Gaussians are gathered to points on a 3D grid and the resulting charge distribution on grid is transformed using FFT to k-space. Poisson equation is solved in k-space for the electrostatic potential and the result is inverse transformed back to real space.

Since the point charges are smeared, the electric potential near the point charges differs from the potential near actual point charges but further away these two potentials converge. *gaussian_width* parameter defines the smearing width and thus what is actually "near". The smaller the *gaussian_width* is the more dense grid of points is needed and the heavier the calculation gets.

Requirements
------------
- Cython
- NumPy

**Cython** must be installed in order to compile the computationally heavy part of the solver. See installition instructions on
`http://docs.cython.org/src/quickstart/install.html <http://docs.cython.org/src/quickstart/install.html>`_
in case you need to install it yourself.

In addition, the solver uses **NumPy**, and the examples use **matplotlib** for plotting and the 'nacl_example.py' uses **ASE (Atomic Simulation Environment)** to create the NaCl structure.

Installition
------------
Put the directory with 'electrostatic_potential_solver.py' and 'gaussians_to_grid.pyx' to your PYTHONPATH environment variable, or run the solver in the directory where the files are.

Usage
-----
See the folder 'examples' for Python script examples with possible use cases of the solver and read the instructions below to understand what happens in them.

The solver is a Python class called 'ElectrostaticPotentialSolver'. To initialize the solver object, use command

solver = ElectrostaticPotentialSolver(atom_coordinates, atom_charges, simulation_cell)

where 'atom_coordinates' is a NumPy array of shape (n_atoms, 3) containing the coordinates of the atoms and 'atom_charges' is an array of shape (n_atoms,) containing the charges of the atoms. 'simulation_cell' is an array of shape (3,) containing the length of the simulation cell in each dimension. The solver needs three computational parameters which determine the accuracy of the solution. These parameters are set using commands

solver.set_parameter('k_cutoff', 3.0)
solver.set_parameter('gaussian_width', 0.5)
solver.set_parameter('gaussian_cutoff', 6.0)

This set of parameter values has been tested to provide accurate results. 'k_cutoff' is the most important parameter, and you can try smaller values to run faster or higher ones to obtain better accuracy. To actually solve the potential, run

solver.solve_potential()

You can obtain the solution on the grid used by the solver by calling

xs, ys, zs, pot_grid = solver.get_data_grid('potential')

In addition to potential, you can get the components of the electric field by substituting 'potential' by 'efield_x', 'efield_y' or 'efield_z'. For convenience, the class provides also a method for obtaining a slice of values at a certain height along one of the axes. It can be used as

xs, ys, pot_slice = solver.get_data_slice('potential', height, normal_direction='z')

where 'height' determines the position along one of the axes at which the slice is taken and 'normal_direction' determines that axis. normal_direction='z' means slice along xy plane, for example.

