import os
import sys
import time
import argparse

import theano
import theano.tensor as T
import numpy as np
import MDAnalysis

import nano_cisc 
import nano_functions 
import grids


"""
This analysis script has been written by Daniel Allen and implements his intrinsic surface representation
of spherical nanoparticles. 

This program can be used to simply visualise the intrinsic surface for abritrary nanoparticles from 
molecular dynamics simulations and also to calculate the radial and intrinsic density of these structures. 

Additionally, the curvature of the resulting surface can be calculated if desired by using the -curves 1 flag upon execution. 
"""

start_time = time.time()

#############################################################################################################################
##################################### First, user line commands are setup ###################################################
#############################################################################################################################
parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-prefix', help='State a prefix for your system to label output files', required=True)
parser.add_argument('-top', help='Topology file', required=True)
parser.add_argument('-traj', help='Trajectory file', required=True)
parser.add_argument('-beta', help='Free parameter for surface fit (default value is 15.0)', required=False)
parser.add_argument('-nanoparticle', help='Mask which selects all atoms in your nanoparticle (required to calculate the centre of mass of your nanoparticle)', required=True)
parser.add_argument('-anchors', help='Mask which selects all anchor atoms used to construct the surface', required=True)
parser.add_argument('-density', help='List of species you wish to calculate the density for e.g:[[group1,mask1],[group2,mask2]]', required=False, type=str, nargs='+')
parser.add_argument('-calcrange', help='Specify calculation range, i.e maximum distance away from micelle centre of mass to include particles in density calculation (default value is 35.0)', required=False)
parser.add_argument('-increment', help='Specify width of spatial intervals (default value is 1.0)', required=False)
parser.add_argument('-XYZsurface', help='Set to 1 if you want to output an XYZ trajectory file of the surface throughout the trajectory (default value is off: 0)', required=False, type=int)
parser.add_argument('-curves', help='Set to 1 if you want to calculate the local Gaussian curvature of the micelle and output to the surface file (default value is off: 0)', required=False, type=int)
args = vars(parser.parse_args())

PREFIX = args['prefix']
top = args['top']
traj = args['traj']

# Initialise universe by loading topology and trajectory files
u = MDAnalysis.Universe(top, traj) 

if args['beta']:
	beta = float(args['beta'])
else:
	beta = 15.0 # Set default value of surface parameter if not specified by user


# Group containing atoms belonging to nanoparticle
nano_particle_mask = args['nanoparticle']
nano_particle = u.select_atoms(nano_particle_mask) 

anchor_mask = args['anchors']
anchors = u.select_atoms(anchor_mask) # Group containing atoms used as anchors

if args['density']:
	print "\nCalculating density...\n"
	int_file_name = "%s-intrinsic-density.dat" % PREFIX
	intrinsic_density_out = open(int_file_name,"w")
	rad_file_name = "%s-radial-density.dat" % PREFIX
	radial_density_out = open(rad_file_name,"w")
	DENSITY = args['density']
	if len(DENSITY) % 2 != 0: # Must input a label AND a mask for each atom group
		print "Odd number of arguments entered for density!!!\nProgram terminated.\n"
		sys.exit(0)
	density=[]

	# set up groups for which the density will be calculated
	intrinsic_density_out.write("# ")
	radial_density_out.write("# ")
	for i in range(len(DENSITY)/2):
		density.append(u.select_atoms(DENSITY[i*2+1]))
		print "I have %d atoms in group: '%s' " % (len(density[i]), DENSITY[i*2])
		intrinsic_density_out.write("%s " % DENSITY[i*2]) # Names of density groups printed to output file headers
		radial_density_out.write("%s " % DENSITY[i*2]) 
	print ""
	intrinsic_density_out.write("\n")
	radial_density_out.write("\n")
else:
	density=[]

# Set calculation range to 35.0 by default if user does not specify value
if args['calcrange']: 
	calcrange = float(args['calcrange'])
else:
	calcrange = 35.0

# Set increment to 1.0 by default if user does not specify value
if args['increment']: 
	targetinc = float(args['increment'])
else:
	targetinc = 1.0

if args['XYZsurface']:
	str = "%s-surface.xyz" % PREFIX
	fvissurf=open(str,"w")
	print "Will write surface to file: %s\n" % str

if args['curves'] == 1:
	curves=1
else:
	curves=0

##############################################################################
######## Define system_ which is the object, of class nanoCISC, ##############
########  which contains all relevant information about your nanoparticle ####
##############################################################################
system_=nano_cisc.nanoCISC(nano_particle, anchors, beta, calcrange, 
                           curves, targetinc, density) 
# initialise system_ as nanoCISC class here ^^^

# If density is being calculated, define grid from grid class
if args['density']:
	grid=grids.grid(system_)


##############################################################################
################ Process trajectory, frame by frame ##########################
##############################################################################

for ts in u.trajectory: # loop through trajectory frames here 
	print "Processing snapshot %d " % (ts.frame)

	# Array for calculating intrinsic density is initialised to {0}
	intrinsic_count=np.zeros( ( np.ceil( np.sqrt(3)*system_.calculation_range).astype(np.int) ,len(system_.density) ), dtype=np.float32) 

	# Array that stores the instantaneous volume of each spatial interval is initialised to {0}
	volume_at_dist=np.zeros( ( np.ceil( np.sqrt(3)*system_.calculation_range ).astype(np.int) ,len(system_.density) ), dtype=np.float32) 

	# Centre of mass position is updated
	nano_functions.update_com(system_)

	# Vectors describing the anchor points are updated 
	nano_functions.update_anchors(system_)  

	# Nanoparticle depth values are updated
	nano_functions.update_surface(system_) 	

	if args['XYZsurface']:
		nano_functions.write_surface(fvissurf,system_) # write micelle surface to xyz file
  
 	if args['density']: 
 		grid.update_volume_estimate(volume_at_dist,system_) # volume estimate is updates for snapshot
		nano_functions.calculate_density(system_,intrinsic_count,volume_at_dist) # calculate density here

	system_.frames_processed += 1
	if system_.frames_processed == 10:
		nano_functions.printintrinsicdensity(intrinsic_density_out,system_)


##################################
##### Print results to files #####
##################################
if args['density']:
	nf.printintrinsicdensity(intrinsic_density_out,system_)


print "Done!!!!!\n"