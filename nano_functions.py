import numpy as np

"""
A series of functions which are used in the main loop
to update quantities relevant to the surface and density calculation
"""

# A function to update the nanoparticle C.O.M. 
# position at each timestep
def update_com(system_):
		for j in range(3):
			system_.mCOM[j] = system_.nano_particle.atoms.center_of_mass()[j] 


 # Function to update anchor vectors (S_i) and magnitudes
def update_anchors(system_):
	for i in range(len(system_.anchors)): 
		for j in range(3):
			system_.ancVECS[i,j] = system_.anchors[i].position[j]-system_.mCOM[j]
		system_.ancVECsize[i] = np.linalg.norm(system_.ancVECS[i,:]) # store magnitude of each S_i vector

		for j in range(3): # scale to unit vectors after the magnitudes have been stored
			system_.ancVECS[i,j] /= system_.ancVECsize[i]	


# Function which calculates depth of nanoparticle surface
# and stores values in an array to look up later
def update_surface(system_):
	for i in range(1,315,4): # loop over theta
		theta = float(i) / 100
		for j in range(1,629,4): # loop over phi
			phi = float(j) / 100
			system_.lookupdepth[ np.floor(i / 4).astype(np.int64), np.floor(j / 4).astype(np.int64) ] = system_.localdepth(theta, phi)
			if system_.curves == 1:
				system_.lookupcurvature[np.floor(i / 4).astype(np.int64),np.floor(j / 4).astype(np.int64)] = system_.localcurvature(theta, phi)


# Function to write the surface to file
# if specified by -XYZsurface flag on execution
def write_surface(outfile,system_):
		outfile.write("12403\n\n") # write number of surface points in one snapshot of xyz file
		refvec = np.zeros(3, dtype = np.float32)
		for i in range(1,314,4): # loop over theta
			theta = float(i) / 100
			for j in range(1,628,4): # loop over phi	
				phi = float(j) / 100
				refvec[0] = np.sin(theta) * np.cos(phi)
				refvec[1] = np.sin(theta) * np.sin(phi)
				refvec[2] = np.cos(theta)
				d = system_.lookupdepth[i / 4, j / 4]
				if system_.curves == 1:
					c = system_.lookupcurvature[i / 4, j / 4]
				if (system_.curves == 1):			
					outfile.write("Sfc %f %f %f %f\n" % (d * refvec[0] + system_.mCOM[0], 
						                                 d * refvec[1] + system_.mCOM[1], 
						                                 d * refvec[2] + system_.mCOM[2], c ))
				else: 
					outfile.write("Sfc %f %f %f\n" % (d * refvec[0], d * refvec[1], d * refvec[2]))


# Function to calculate the radial and intrinsic 
# densities of the nanoparticle
def calculate_density(system_, intcount, voldist):
	for i in range(len(system_.density)):
		# loop over each density group
		for j in range(len(system_.density[i])):
			# loop over all atoms within density group i	
			if (abs(system_.density[i][j].position[0] - system_.mCOM[0] ) < system_.calculation_range 
				 and abs(system_.density[i][j].position[1] - system_.mCOM[1] ) < system_.calculation_range 
				   and abs(system_.density[i][j].position[2] - system_.mCOM[2]) < system_.calculation_range ):
					# if atom density[i][j] is within the grid that was specified then include it in the calculation
				th = (np.rint(100 * system_.calctheta(system_.density[i][j].position - system_.mCOM)) / 4).astype(np.int64)
				ph = system_.calcphi(system_.density[i][j].position-system_.mCOM)
				if ph < 0.00000:
					ph += 2 * np.pi
				ph = (np.rint(100 * ph) / 4).astype(np.int64)
				intcount[ np.rint((1 / system_.target_increment) * (np.linalg.norm(system_.density[i][j].position-system_.mCOM)
					     - system_.lookupdepth[th,ph])).astype(np.int64), i ] += 1.0
				system_.radialdensity[np.rint((1 / system_.target_increment) * (np.linalg.norm(system_.density[i][j].position
					     - system_.mCOM))).astype(np.int64), i ] += 1.0

		for j in range(len(intcount[:,i])):
			if voldist[j] > 0:
				system_.intrinsicdensity[j,i] += intcount[j,i] / voldist[j]


def printintrinsicdensity(intdensityout,system_):
	for i in range(-20,31):
		intdensityout.write("%f " % (system_.target_increment * float(i)))
		for j in range(len(system_.density)):
			intdensityout.write("%f " % (system_.intrinsicdensity[i,j] / float(system_.frames_processed)) ) 
		intdensityout.write("\n")
	intdensityout.write("\n")

#def printradialdensity():
#	for i in range(0,100):
#			raddensityout.write("%f " % (targetinc*float(i)) )
#			for j in range(len(density)):
#				raddensityout.write("%f " % (radialdensity[i,j]/(float(count)*( (4*np.pi/3)*(i*targetinc+0.5*targetinc)**3 - (4*np.pi/3)*(i*targetinc-0.5*targetinc)**3 ) ) ) ) 
#			raddensityout.write("\n")
#		raddensityout.write("\n")
#	raddensityout.write("\n")