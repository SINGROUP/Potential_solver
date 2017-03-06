# -*- coding: utf-8 -*-
#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

from ase import Atoms
from ase.lattice.cubic import SimpleCubicFactory
from ase.visualize import view

from electrostatic_potential_solver import ElectrostaticPotentialSolver

# To prevent a layer of element one on one side, and a layer of
# element two on the other side, NaCl is based on SimpleCubic instead
# of on FaceCenteredCubic
class NaClFactory(SimpleCubicFactory):
    "A factory for creating NaCl (B1, Rocksalt) lattices."

    bravais_basis = [[0, 0, 0], [0, 0, 0.5], [0, 0.5, 0], [0, 0.5, 0.5],
                   [0.5, 0, 0], [0.5, 0, 0.5], [0.5, 0.5, 0],
                   [0.5, 0.5, 0.5]]
    element_basis = (0, 1, 1, 0, 1, 0, 0, 1)

def charge_by_elem(elem):
    if elem == 'Na':
        return 0.5
    elif elem == 'Cl':
        return -0.5
    else:
        return 0.0


# Parameters
cell_reps = [5,2,5]
a = 5.73
vacuum = 15.0
slice_z = 1.0


# Create NaCl slab
NaCl = NaClFactory();
nacl_slab = NaCl(directions=[[1,0,0],[0,1,0],[0,0,1]],
                size=(cell_reps[0],cell_reps[1],cell_reps[2]),
                symbol=['Na','Cl'], pbc=(1,0,1), latticeconstant=a)
nacl_slab.translate([0,vacuum,0])
nacl_slab.cell[1,1] = nacl_slab.cell[1,1]-0.5*a+2*vacuum


# Get atom coordinates and charges
sim_cell = np.array([nacl_slab.cell[0,0], nacl_slab.cell[1,1], nacl_slab.cell[2,2]])
charges = np.array(map(charge_by_elem, nacl_slab.get_chemical_symbols()))

# Solve electrostatic potential
solver = ElectrostaticPotentialSolver(nacl_slab.get_positions(), charges, sim_cell)
solver.set_parameter('k_cutoff', 2.0)
solver.set_parameter('gaussian_width', 0.5)
solver.set_parameter('gaussian_cutoff', 6.0)
solver.solve_potential()

# Plot charge
xs, zs, charge_slice = solver.get_data_slice('charge', vacuum+1.5*a+slice_z, normal_direction='y')
plt.figure()
plt.contourf(xs, zs, charge_slice.T, 50)
plt.axes().set_aspect('equal')
plt.title(u'Charge density, z = {:.2f} Å'.format(slice_z), size=16)
plt.xlabel(u'x (Å)', size=14)
plt.ylabel(u'y (Å)', size=14)
cbar = plt.colorbar()
cbar.ax.set_ylabel(u'$\\rho$ (e/Å$^{3}$)', size=14)

# Plot potential
xs, zs, pot_slice = solver.get_data_slice('potential', vacuum+1.5*a+slice_z, normal_direction='y')
plt.figure()
plt.contourf(xs, zs, pot_slice.T, 50)
plt.axes().set_aspect('equal')
plt.title(u'Electrostatic potential, z = {:.2f} Å'.format(slice_z), size=16)
plt.xlabel(u'x (Å)', size=14)
plt.ylabel(u'y (Å)', size=14)
cbar = plt.colorbar()
cbar.ax.set_ylabel(u'$\\Phi$ (V)', size=14)

# Plot electric field
xs, zs, efield_x_slice = solver.get_data_slice('efield_x', vacuum+1.5*a+slice_z, normal_direction='y')
xs, zs, efield_y_slice = solver.get_data_slice('efield_y', vacuum+1.5*a+slice_z, normal_direction='y')
xs, zs, efield_z_slice = solver.get_data_slice('efield_z', vacuum+1.5*a+slice_z, normal_direction='y')
plt.figure()
plt.contourf(xs, zs, efield_y_slice.T, 50)
cbar = plt.colorbar()
cbar.ax.set_ylabel(u'$E_z$ (V/Å)', size=14)
plt.quiver(xs, zs, efield_x_slice.T, efield_z_slice.T, scale=40.0)
plt.axes().set_aspect('equal')
plt.title(u'Electric field, z = {:.2f} Å'.format(slice_z), size=16)
plt.xlabel(u'x (Å)', size=14)
plt.ylabel(u'y (Å)', size=14)

plt.show()
