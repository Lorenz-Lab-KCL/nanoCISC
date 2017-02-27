import theano
import theano.tensor as T
import numpy as np 

class nanoCISC : 
	"""
	Class which we will use to handle all of the various objects used in analysis.
	The nanoCISC object inherits attributes passed from user on the command line
	and contains functions that we will use throughout analysis.
	 """
	def __init__(self, nanoparticle, anchors, beta, calcrange, curves, targetinc, density): 
		"""
		Define and setup all the attributes of a nanoCISC object
		"""
		self.nano_particle = nanoparticle
		self.anchors = anchors
		self.beta = beta
		self.calculation_range = calcrange
		self.curves = curves
		self.target_increment = targetinc
		self.density = density
  		self.frames_processed = 0 # Count how many frames have been processed during analysis
  		density_array_size = np.ceil(3 * (self.calculation_range)).astype(np.int)
  		# Array to store intrinsic density
		self.intrinsicdensity = np.zeros((density_array_size ,len(self.density)), dtype = np.float32)
		# Array to store radial density
		self.radialdensity = np.zeros((density_array_size, len(self.density)), dtype = np.float32) 
		self.mCOM = np.zeros(3, dtype = np.float32) # position vector of nanoparticle centre of mass
		self.ancVECS = np.zeros((len(self.anchors), 3), dtype = np.float32) # array to store unit vector components of S_i
		self.ancVECsize = np.zeros(len(self.anchors), dtype = np.float32) # vector of magnitudes of S_i

		# Array to store values of depth at different angles
		self.lookupdepth=np.zeros((79, 158), dtype = np.float32) 

		# Array to store local curvature values at different angles
		self.lookupcurvature=np.zeros((79, 158), dtype = np.float32)  

		# specify symbolic variables for theano 
		THETA=T.dscalar() # specify that angles and BETA are scalars
		PHI=T.dscalar()
		BETA = theano.shared(value = self.beta, name = 'BETA', borrow = True)
		MAG = theano.shared(value = self.ancVECsize, name = 'MAG', borrow = True) # vector that symbalises the magitudes of S_i
		Si = theano.shared(value = self.ancVECS, name = 'Si', borrow = True) # matrix that symbalises S_i vectors
		COM = theano.shared(value = self.mCOM, name = 'COM', borrow = True)

		# Define the angular resolution for which the surface is stored
		self.angle_increment = 4
		self.span_theta = 315
		self.span_phi = 629 

		############################################################################
		################### DEFINE SOME FUNCTIONS THAT WE WILL NEED LATER ##########
		############################################################################
		print "Setting up and compiling intrinsic surface function... sit tight.\n"

		refX=T.sin(THETA)*T.cos(PHI) # expressions for unit vector components of direction vector
		refY=T.sin(THETA)*T.sin(PHI)
		refZ=T.cos(THETA)

		###################################################################################
		################## define DEPTH and CALCTHETA/CALCPHI functions ###################
		###################################################################################
		depth = ((MAG * T.exp(-BETA * T.arccos(T.clip(T.dot(Si, [refX,refY,refZ]), -1.0, 1.0)) ** 2 )).sum() / 
			       (T.exp(-BETA * T.arccos(T.clip(T.dot(Si, [refX,refY,refZ]),-1.0,1.0)) ** 2  )).sum() )

		self.DEPTH = theano.function([THETA,PHI], depth) # depth function is compiled here

		pos = depth * [refX,refY,refZ] # x,y,z coords of surface at (THETA,PHI)
		POS = theano.function([THETA,PHI],pos) # function that returns position of the surface is compiled here
		print "Done.\n"

		VEC = T.vector()
		calctheta = T.arccos(T.clip(VEC[2] / T.sqrt(VEC[0] ** 2 + VEC[1] ** 2 + VEC[2] ** 2), -1.0, 1.0))
		self.CALCTHETA = theano.function([VEC], calctheta) # function is compiled to calculate theta
		calcphi = T.arctan2(VEC[1], VEC[0]) 
		self.CALCPHI = theano.function([VEC], calcphi) # function is compiled to calculate phi

		if self.curves == 1: # if curvature is switched on, i.e. -curves 1, then the curvature function is defined and compiled here

			#########################################################################################
			############################## CURVATURE FUNCTION #######################################
			#########################################################################################

			print "Setting up and compiling curvature function... sit tight, this could take a moment...\n"
			# set up derivatives for curvature function
			gT = [T.grad(pos[0], THETA), T.grad(pos[1], THETA), T.grad(pos[2], THETA)]
			gP = [T.grad(pos[0], PHI), T.grad(pos[1], PHI), T.grad(pos[2], PHI)]
			# SECOND DERIVATIVES
			gTT = [T.grad(gT[0], THETA), T.grad(gT[1], THETA), T.grad(gT[2], THETA)] 
			gPP = [T.grad(gP[0], PHI), T.grad(gP[1], PHI), T.grad(gP[2], PHI)] 
			gTP = [T.grad(gT[0], PHI), T.grad(gT[1], PHI), T.grad(gT[2], PHI)]

			E = T.dot(gT,gT) # calculate first fundamental form
			F = T.dot(gT,gP)
			G = T.dot(gP,gP) 

			# calculate second fundamental form
			# calculate normal vector to surface
			n = ( [gT[1] * gP[2] - gT[2] * gP[1], gT[2] * gP[0] - gT[0] * gP[2], gT[0] * gP[1] - gT[1] * gP[0]] 
			  /  T.sqrt(T.dot([gT[1] * gP[2] - gT[2] * gP[1], gT[2] * gP[0] - gT[0] * gP[2], gT[0] * gP[1] 
			  - gT[1] * gP[0]] , [gT[1] * gP[2] - gT[2] * gP[1], gT[2] * gP[0] - gT[0] * gP[2], gT[0] * gP[1] - gT[1] * gP[0]]))
			  )

			L = T.dot(gTT, n)
			M = T.dot(gTP, n)
			N = T.dot(gPP, n)

			self.CURVATURE = theano.function([THETA,PHI], (L * N - M ** 2)/( E * G - F ** 2)) 
			print "Done.\n"	

	# define functions to return values from binary theano functions
	def localdepth(self,a,b):
		return self.DEPTH(a,b) 

	def localcurvature(self,a,b):
		return self.CURVATURE(a,b)

	def calctheta(self,a):
		return self.CALCTHETA(a)

	def calcphi(self,a):
		return self.CALCPHI(a)


	# A function to update the nanoparticle C.O.M. 
	# position at each timestep
	def update_com(self):
		for j in range(3):
			self.mCOM[j] = self.nano_particle.atoms.center_of_mass()[j] 


	 # Function to update anchor vectors (S_i) and magnitudes
	def update_anchors(self):
		for i in range(len(self.anchors)): 
			for j in range(3):
				self.ancVECS[i,j] = self.anchors[i].position[j] - self.mCOM[j]
			self.ancVECsize[i] = np.linalg.norm(self.ancVECS[i,:]) # store magnitude of each S_i vector

			for j in range(3): # scale to unit vectors after the magnitudes have been stored
				self.ancVECS[i,j] /= self.ancVECsize[i]


	# Function which calculates depth of nanoparticle surface
	# and stores values in an array to look up later
	def update_surface(self):
		for i in range(1, self.span_theta, self.angle_increment): # loop over theta
			theta = float(i) / 100
			for j in range(1, self.span_phi, self.angle_increment): # loop over phi
				phi = float(j) / 100
				self.lookupdepth[ np.floor(i / self.angle_increment).astype(np.int64), np.floor(j / self.angle_increment).astype(np.int64) ] = self.localdepth(theta, phi)
				if self.curves == 1:
					self.lookupcurvature[np.floor(i / self.angle_increment).astype(np.int64),np.floor(j / self.angle_increment).astype(np.int64)] = self.localcurvature(theta, phi)


	# Function to write the surface to file
	# if specified by -XYZsurface flag on execution
	def write_surface(self,outfile):
		outfile.write("12403\n\n") # write number of surface points in one snapshot of xyz file
		refvec = np.zeros(3, dtype = np.float32)
		for i in range(1, self.span_theta, self.angle_increment): # loop over theta
			theta = float(i) / 100
			for j in range(1, self.span_phi, self.angle_increment): # loop over phi	
				phi = float(j) / 100
				refvec[0] = np.sin(theta) * np.cos(phi)
				refvec[1] = np.sin(theta) * np.sin(phi)
				refvec[2] = np.cos(theta)
				d = self.lookupdepth[i / self.angle_increment, j / self.angle_increment]
				if self.curves == 1:
					c = self.lookupcurvature[i / self.angle_increment, j / self.angle_increment]
				if (self.curves == 1):			
					outfile.write("Sfc %f %f %f %f\n" % (d * refvec[0] + self.mCOM[0], 
						                                 d * refvec[1] + self.mCOM[1], 
						                                 d * refvec[2] + self.mCOM[2], c ))
				else: 
					outfile.write("Sfc %f %f %f\n" % (d * refvec[0], d * refvec[1], d * refvec[2]))


	# Function to calculate the radial and intrinsic 
	# densities of the nanoparticle
	def calculate_density(self, intcount, voldist):
		for i in range(len(self.density)):
			# loop over each density group
			for j in range(len(self.density[i])):
				# loop over all atoms within density group i	
				if (abs(self.density[i][j].position[0] - self.mCOM[0] ) < self.calculation_range 
				 	and abs(self.density[i][j].position[1] - self.mCOM[1] ) < self.calculation_range 
				   		and abs(self.density[i][j].position[2] - self.mCOM[2]) < self.calculation_range ):
				# if atom density[i][j] is within the grid that was specified then include it in the calculation
					th = (np.rint(100 * self.calctheta(self.density[i][j].position - self.mCOM)) / self.angle_increment).astype(np.int64)
					ph = self.calcphi(self.density[i][j].position-self.mCOM)
					if ph < 0.00000:
						ph += 2 * np.pi
					ph = (np.rint(100 * ph) / self.angle_increment).astype(np.int64)
					intcount[ np.rint((1 / self.target_increment) * (np.linalg.norm(self.density[i][j].position - self.mCOM)
					     	 - self.lookupdepth[th,ph])).astype(np.int64), i ] += 1.0
					self.radialdensity[np.rint((1 / self.target_increment) * (np.linalg.norm(self.density[i][j].position
					         - self.mCOM))).astype(np.int64), i ] += 1.0

			for j in range(len(intcount[:,i])):
				if voldist[j] > 0:
					self.intrinsicdensity[j,i] += intcount[j,i] / voldist[j]


	def print_intrinsic_density(self,intdensityout):
		for i in range(-20,31): # Print density for range -20-30 Angstroms
			intdensityout.write("%f " % (self.target_increment * float(i)))
			for j in range(len(self.density)):
				intdensityout.write("%f " % (self.intrinsicdensity[i,j] / float(self.frames_processed)) ) 
			intdensityout.write("\n")
		intdensityout.write("\n")


	def print_radial_density(self):
		for i in range(0,100): # Print density for range 0-100 Angstroms 
			raddensityout.write("%f " % (targetinc*float(i)) )
			for j in range(len(density)):
				raddensityout.write("%f " % (radialdensity[i,j] / (float(count) * ((4 * np.pi / 3) * (i * targetinc + 0.5 * targetinc) ** 3 
											- (4 * np.pi / 3) * (i * targetinc - 0.5 * targetinc) ** 3)))) 
			raddensityout.write("\n")
		raddensityout.write("\n")