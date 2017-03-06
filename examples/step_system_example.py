# -*- coding: utf-8 -*-
#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from electrostatic_potential_solver import ElectrostaticPotentialSolver

filename = 'old_whole_step_charges.xyz'
sim_cell = np.array([42.616, 73.816, 54.424])
#filename = 'triangle_small_whole_step_charges.xyz'
#sim_cell = np.array([42.616, 55.362, 57.424])
#filename = '20x10x3_triangle_big_charges.xyz'
#sim_cell = np.array([106.54, 92.27, 43.068])

slice_z = 1.0

# Load geometry and charges
data = np.loadtxt(filename, skiprows=2, usecols=(1,2,3,4))
atom_coordinates = data[:, 0:3]
atom_charges = data[:, 3]
surf_z = atom_coordinates[:, 2].max()

# Solve electrostatic potential
solver = ElectrostaticPotentialSolver(atom_coordinates, atom_charges, sim_cell)
solver.set_parameter('k_cutoff', 3.0) # optimal: 3.0
solver.set_parameter('gaussian_width', 0.5) # optimal: 0.5
solver.set_parameter('gaussian_cutoff', 6.0) # optimal: 6.0
solver.solve_potential()

# If you want to get the complete data grid instead of just a slice, use get_data_grid method. For example:
# xs, ys, zs, pot_grid = solver.get_data_grid('potential')

# Plot charge distribution
xs, ys, charge_slice = solver.get_data_slice('charge', surf_z+slice_z, normal_direction='z')
plt.figure()
plt.contourf(xs, ys, charge_slice.T, 50)
plt.axes().set_aspect('equal')
plt.title(u'Charge density, z = {:.2f} Å'.format(slice_z), size=16)
plt.xlabel(u'x (Å)', size=14)
plt.ylabel(u'y (Å)', size=14)
cbar = plt.colorbar()
cbar.ax.set_ylabel(u'$\\rho$ (e/Å$^{3}$)', size=14)

# Plot potential
xs, ys, pot_slice = solver.get_data_slice('potential', surf_z+slice_z, normal_direction='z')
plt.figure()
plt.contourf(xs, ys, pot_slice.T, 50)
plt.axes().set_aspect('equal')
plt.title(u'Electrostatic potential, z = {:.2f} Å'.format(slice_z), size=16)
plt.xlabel(u'x (Å)', size=14)
plt.ylabel(u'y (Å)', size=14)
cbar = plt.colorbar()
cbar.ax.set_ylabel(u'$\\Phi$ (V)', size=14)

# Plot electric field
xs, ys, efield_x_slice = solver.get_data_slice('efield_x', surf_z+slice_z, normal_direction='z')
xs, ys, efield_y_slice = solver.get_data_slice('efield_y', surf_z+slice_z, normal_direction='z')
xs, ys, efield_z_slice = solver.get_data_slice('efield_z', surf_z+slice_z, normal_direction='z')
plt.figure()
plt.contourf(xs, ys, efield_z_slice.T, 50)
cbar = plt.colorbar()
cbar.ax.set_ylabel(u'$E_z$ (V/Å)', size=14)
plt.quiver(xs[::2], ys[::2], efield_x_slice[::2, ::2].T, efield_y_slice[::2, ::2].T)
plt.axes().set_aspect('equal')
plt.title(u'Electric field, z = {:.2f} Å'.format(slice_z), size=16)
plt.xlabel(u'x (Å)', size=14)
plt.ylabel(u'y (Å)', size=14)
filename = fig_filename_base + '_efield_z{:.2f}.png'.format(slice_z)
plt.savefig(filename)
filename = fig_filename_base + '_efield_z{}.png'.format(slice_z)
plt.savefig(filename, dpi=200)

plt.show()
