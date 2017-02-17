import theano
import theano.tensor as T
import numpy as np 

class nanoCISC :
	"""
	Class which we will use to handle all of the various objects used in analysis.
	The nanoCISC object inherits attributes passed from user on the command line
	 """
	def __init__(self, nanoparticle,anchors,beta,calcrange,curves,targetinc,density): 
		self.nano_particle = nanoparticle
		self.anchors = anchors
		self.beta = beta
		self.calculation_range = calcrange
		self.curves = curves
		self.target_increment = targetinc
		self.density = density
  		self.frames_processed = 0

		self.intrinsicdensity = np.zeros((np.ceil(np.sqrt(3) * self.calculation_range).astype(np.int) ,len(self.density)), dtype = np.float32)
		self.radialdensity=np.zeros((np.ceil(np.sqrt(3) * self.calculation_range).astype(np.int) ,len(self.density)), dtype = np.float32) # calculate radial density too
		
		self.mCOM=np.zeros(3, dtype=np.float32) # position vector of nanoparticle centre of mass
		self.ancVECS=np.zeros((len(self.anchors), 3), dtype = np.float32) # array to store unit vector components of S_i
		self.ancVECsize=np.zeros(len(self.anchors), dtype = np.float32) # vector of magnitudes of S_i

		# array to store values of depth at different angles
		self.lookupdepth=np.zeros((int(np.floor(79)), int(np.floor(158))), dtype = np.float32) 

		# array to store local curvature values at different angles
		self.lookupcurvature=np.zeros((int(np.floor(79)), int(np.floor(158))), dtype = np.float32)  

		# specify symbolic variables for theano 
		THETA=T.dscalar() # specify that angles and BETA are scalars
		PHI=T.dscalar()
		BETA = theano.shared(value = self.beta, name = 'BETA', borrow = True)
		MAG = theano.shared(value = self.ancVECsize, name = 'MAG', borrow = True) # vector that symbalises the magitudes of S_i
		Si = theano.shared(value = self.ancVECS, name = 'Si', borrow = True) # matrix that symbalises S_i vectors
		COM = theano.shared(value = self.mCOM, name = 'COM', borrow = True)

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
		calctheta = T.arccos( T.clip(VEC[2] / T.sqrt(VEC[0] ** 2 + VEC[1] ** 2 + VEC[2] ** 2), -1.0, 1.0))
		self.CALCTHETA = theano.function([VEC],calctheta) # function is compiled to calculate theta
		calcphi = T.arctan2(VEC[1],VEC[0]) 
		self.CALCPHI = theano.function([VEC],calcphi) # function is compiled to calculate phi

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