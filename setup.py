import os
import sys

import MDAnalysis

import grids

class initialise_user_input :
	"""
	Class which takes care of processing the arguments
	passed in on the command line and setting up the 
	calculations.
	"""
	def __init__(self,args):

		top = args['top']
		traj = args['traj']
		self.u = MDAnalysis.Universe(top, traj)
		self.PREFIX = args['prefix']

		nano_particle_mask = args['nanoparticle']
		self.nano_particle = self.u.select_atoms(nano_particle_mask)

		anchor_mask = args['anchors']
		self.anchors = self.u.select_atoms(anchor_mask) # Group containing atoms used as anchors

		if args['beta']:
			self.beta = float(args['beta'])
		else:
			self.beta = 15.0 # Set default value of surface parameter if not specified by user

		if args['density']:
			
			print "\nCalculating density...\n"

			intrinsic_file_name = "%s-intrinsic-density.dat" % self.PREFIX
			self.f_intrinsic_density_out = open(intrinsic_file_name,"w")

			radial_file_name = "%s-radial-density.dat" % self.PREFIX
			self.f_radial_density_out = open(radial_file_name,"w")
			
			DENSITY = args['density']
			if len(DENSITY) % 2 != 0: # Must input a label AND a mask for each atom group
				print "Odd number of arguments entered for density!!!\nProgram terminated.\n"
				sys.exit(0)
			self.density=[]

			# Write header to density files (atom group labels) 
			self.f_intrinsic_density_out.write("# ")
			self.f_radial_density_out.write("# ") 

			# set up groups for which the density will be calculated
			for i in range(len(DENSITY)/2):
				self.density.append(self.u.select_atoms(DENSITY[i*2+1]))
				print "I have %d atoms in group: '%s' " % (len(self.density[i]), DENSITY[i*2])
				self.f_intrinsic_density_out.write("%s " % DENSITY[i*2]) # Names of density groups printed to output file headers
				self.f_radial_density_out.write("%s " % DENSITY[i*2]) 
			print ""
			self.f_intrinsic_density_out.write("\n")
			self.f_radial_density_out.write("\n")

			#grid=grids.grid(system_) # Build grid to calculate volume of spatial regions

		else:
			self.density=[]

		# Set calculation range to 35.0 by default if user does not specify value
		if args['calcrange']: 
			self.calcrange = float(args['calcrange'])
		else:
			self.calcrange = 35.0

		# Set density increment to 1.0 by default if user does not specify value
		if args['increment']: 
			self.targetinc = float(args['increment'])
		else:
			self.targetinc = 1.0

		if args['XYZsurface']:
			str = "%s-surface.xyz" % PREFIX
			self.f_visualise_surface=open(str,"w")
			print "Will write surface to file: %s\n" % str

		if args['curves'] == 1:
			self.curves=1
		else:
			self.curves=0