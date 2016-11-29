import os
import sys
import numpy as np
import theano
import theano.tensor as T
import MDAnalysis
import time
import argparse
start_time = time.time()


#############################################################################################################################
##################################### FIRST USER LINE COMMANDS ARE SETUP ####################################################
#############################################################################################################################

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-prefix', help='State a prefix for your system to label output files', required=True)
parser.add_argument('-top', help='Topology file', required=True)
parser.add_argument('-traj', help='Trajectory file', required=True)
parser.add_argument('-beta', help='Free parameter for surface fit (default value is 15.0)', required=False)
parser.add_argument('-micelle', help='Mask which selects all atoms in your micelle (required to calculate the centre of mass of your micelle)', required=True)
parser.add_argument('-anchors', help='Mask which selects all anchor atoms used to construct the surface', required=True)
parser.add_argument('-density', help='List of species you wish to calculate the density for e.g:[[group1,mask1],[group2,mask2]]', required=False, type=str, nargs='+')
parser.add_argument('-calcrange', help='Specify calculation range, i.e maximum distance away from micelle centre of mass to include particles in density calculation (default value is 35.0)', required=False)
parser.add_argument('-increment', help='Specify width of spatial intervals (default value is 1.0)', required=False)
parser.add_argument('-XYZsurface', help='Set to 1 if you want to output an XYZ trajectory file of the surface throughout the trajectory (default value is off: 0)', required=False, type=int)
parser.add_argument('-curves', help='Set to 1 if you want to calculate the local Gaussian curvature of the micelle and output to the surface file (default value is off: 0)', required=False, type=int)
args = vars(parser.parse_args())

PREFIX=args['prefix']
#print "\nSystem prefix set to %s\n" % PREFIX

top=args['top']
traj=args['traj']
u = MDAnalysis.Universe(top, traj) # LOAD TOPOLOGY AND TRAJECTORY

if args['beta']:
	beta=float(args['beta'])
else:
	beta=15.0 # SET DEFAULT VALUE OF SURFACE PARAMETER IF NOT SPECIFIED BY USER

micellemask=args['micelle']
micelle=u.select_atoms(micellemask) # GROUP CONTAINING ATOMS BELONGING TO MICELLE

anchortype="type 8"
anchors=u.select_atoms(anchortype) # GROUP CONTAINING ATOMS USED AS ANCHORS

if args['density']:
	print "\nCalculating density...\n"
	intoutstr="%s-intrinsic-density.dat" % PREFIX
	intdensityout=open(intoutstr,"w")

	radoutstr="%s-radial-density.dat" % PREFIX
	raddensityout=open(radoutstr,"w")
	
	DENSITY=args['density']
	if len(DENSITY) % 2 != 0:
		print "Odd number of arguments entered for density!!!\nProgram terminated.\n"
		sys.exit(0)

	density=[]

	# set up groups for which the density will be calculated
	intdensityout.write("# ")
	raddensityout.write("# ")
	for i in range(len(DENSITY)/2):
		density.append(u.select_atoms(DENSITY[i*2+1]))
		print "I have %d atoms in group: '%s' " % (len(density[i]), DENSITY[i*2])
		intdensityout.write("%s " % DENSITY[i*2]) # Names of density groups printed to output file headers
		raddensityout.write("%s " % DENSITY[i*2]) 
	print ""
	intdensityout.write("\n")
	raddensityout.write("\n")

if args['calcrange']:
	calcrange=float(args['calcrange'])
else:
	calcrange=35.0

if args['increment']:
	targetinc=float(args['increment'])
else:
	targetinc=1.0

if args['XYZsurface']:
	str="%s-surface.xyz" % PREFIX
	fvissurf=open(str,"w")
	print "Will write surface to file: %s\n" % str

if args['curves']==1:
	curves=1
else:
	curves=0


###################################################################################################################
################################## NOW SETUP SYMBOLIC AND SHARED VARIABLES FOR THEANO #############################
###################################################################################################################
CR=theano.shared(value=calcrange, name='CR', borrow=True)
mCOM=np.zeros(3, dtype=np.float32) # POSITION VECTOR OF MICELLE CENTRE OF MASS
ancVECS=np.zeros((len(anchors),3), dtype=np.float32) # array to store unit vector components of S_i
ancVECsize=np.zeros(len(anchors), dtype=np.float32) # vector of magnitudes of S_i

intrinsicdensity=np.zeros((600,len(density)), dtype=np.float32)
radialdensity=np.zeros((600,len(density)), dtype=np.float32) # calculate radial density too

# specify symbolic variables for theano 
index=T.iscalar()
THETA=T.dscalar() # specify that angles and BETA are scalars
PHI=T.dscalar()
BETA = theano.shared(value=beta, name='BETA', borrow=True)
MAG = theano.shared(value=ancVECsize, name='MAG', borrow=True) # vector that symbalises the magitudes of S_i
Si = theano.shared(value=ancVECS, name='Si', borrow=True) # matrix that symbalises S_i vectors
COM = theano.shared(value=mCOM, name='COM', borrow=True)

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
depth = (MAG*T.exp(-BETA*T.arccos( T.clip(T.dot(Si,[refX,refY,refZ]),-1.0,1.0) )**2 )).sum() / (T.exp(-BETA*T.arccos( T.clip(T.dot(Si,[refX,refY,refZ]),-1.0,1.0))**2  )).sum()
DEPTH=theano.function([THETA,PHI], depth) # depth function is compiled here
pos=depth*[refX,refY,refZ] # x,y,z coords of surface at (THETA,PHI)
POS=theano.function([THETA,PHI],pos) # function that returns position of the surface is compiled here
print "Done.\n"

VEC=T.vector()
calctheta=T.arccos( T.clip( VEC[2] / T.sqrt( VEC[0]**2 + VEC[1]**2 + VEC[2]**2),-1.0,1.0 ) )
CALCTHETA=theano.function([VEC],calctheta) # function is compiled to calculate theta
calcphi=T.arctan2(VEC[1],VEC[0]) 
CALCPHI=theano.function([VEC],calcphi) # function is compiled to calculate phi


######################################################################################################################
########################## FUNCTIONS TO CALCULATE AND VISUALISE MICELLE INTRINSIC SURFACE ############################
######################################################################################################################
lookupdepth=np.zeros((int(np.floor(79)),int(np.floor(158))), dtype=np.float32) # array to store values of depth at different angles
if curves==1:
	lookupcurvature=np.zeros((int(np.floor(79)),int(np.floor(158))), dtype=np.float32)

def updateCOM(mCOM):
	for j in range(3):
		mCOM[j] = micelle.atoms.center_of_mass()[j] # update C.O.M. position

def updateanchors(ancVECS,ancVECsize,mCOM):
	# UPDATE S_i VECTORS AND MAGNITUDES TO ARRAYS
	for i in range(len(anchors)): 
		for j in range(3):
			ancVECS[i,j] = anchors[i].position[j]-mCOM[j]
		ancVECsize[i]=np.linalg.norm(ancVECS[i,:]) # store magnitude

		for j in range(3): # scale to unit vectors after storing magnitudes
			ancVECS[i,j] /= ancVECsize[i]	

def updateSURFACE():
		for i in range(1,315,4): # loop over theta
			theta=float(i)/100
			for j in range(1,629,4): # loop over phi
				phi=float(j)/100
				lookupdepth[np.floor(i/4).astype(np.int64),np.floor(j/4).astype(np.int64)] = DEPTH(theta,phi)
				if curves==1:
					lookupcurvature[np.floor(i/4).astype(np.int64),np.floor(j/4).astype(np.int64)] = CURVATURE(theta,phi)


if args['XYZsurface']:
	def writeSURFACEandCURVATURE(outfile,curve):
		outfile.write("12403\n\n" ) # write number of points in one snapshot of xyz file
		refvec=np.zeros(3,dtype=np.float32)
		for i in range(1,314,4):
			theta=float(i)/100
			for j in range(1,628,4): # loop over phi	
				phi=float(j)/100
				refvec[0]=np.sin(theta)*np.cos(phi)
				refvec[1]=np.sin(theta)*np.sin(phi)
				refvec[2]=np.cos(theta)
				d=lookupdepth[i/4,j/4]
				if curves==1:
					c=lookupcurvature[i/4,j/4]

				if (curves==1):			
					outfile.write("%f %f %f %f\n" % (d*refvec[0], d*refvec[1], d*refvec[2], c ) )
				else: 
					outfile.write("%f %f %f\n" % (d*refvec[0], d*refvec[1], d*refvec[2]) )

###########################################################################################################
######################### BUILD FUNCTION THAT CALCULATES VOLUME OF SPATIAL INTERVALS ######################
###########################################################################################################
print "Setting up intrinsic density calculation... sit tight.\n"

nBOXS=int(np.ceil(calcrange*2/targetinc))
NB=theano.shared(value=nBOXS, name='NB', borrow=True)

sBOXS=(calcrange*2)/nBOXS
SB=theano.shared(value=sBOXS, name='SB', borrow=True)

distindices=np.zeros((nBOXS**3,3),dtype=np.float32) # 3d array will store distance to surface for grid points
DISTINDICES = theano.shared(value=distindices, name='DISTINDICES', borrow=True)

count = 0
for x in range(nBOXS):
	for y in range(nBOXS):
		for z in range(nBOXS):
			distindices[count,:]=[x,y,z]
			count += 1

DISTINDICES = theano.shared(value=distindices, name='DISTINDICES', borrow=True)
gridvecs = ( SB/2 - CR + SB*DISTINDICES )
UPDATEGRIDVECS = theano.function([], gridvecs)

up=UPDATEGRIDVECS() # array stores coords of grid points
griddists=np.zeros(len(up[:,0]), dtype=float)
gridangles=np.zeros((len(up[:,0]),2), dtype=np.int64)
for i in range(len(up[:,0])):
	griddists[i]=np.linalg.norm(up[i,:])
	gridangles[i,0] = np.rint(CALCTHETA(up[i,:])*25).astype(np.int64) 
	gridangles[i,1] = CALCPHI(up[i,:])
	if gridangles[i,1] < 0.0:
		gridangles[i,1] += 2*np.pi
		gridangles[i,1] = np.rint(gridangles[i,1]*25).astype(np.int64) 
	else:
		gridangles[i,1] = np.rint(gridangles[i,1]*25).astype(np.int64)

def updateVOLESTIMATE(array):
		for i in range(len(up[:,0])):
			array[ int(np.rint( ( (1/targetinc)*(griddists[i] - lookupdepth[ gridangles[i,0], gridangles[i,1] ] ) ) ) )  ] += sBOXS**3

	

###########################################################################################################
######################### BUILD FUNCTION THAT CALCULATES INTRINSIC DENSITY ################################
###########################################################################################################

def calcdensity(intcount,voldist):
	for i in range(len(density)):
			for j in range(len(density[i])):	
				if abs(density[i][j].position[0]-mCOM[0]) < calcrange and abs(density[i][j].position[1]-mCOM[1]) < calcrange and abs(density[i][j].position[2]-mCOM[2]) < calcrange:
					th=(np.rint(100*CALCTHETA(density[i][j].position-mCOM))/4).astype(np.int64)
					ph=CALCPHI(density[i][j].position-mCOM)
			
					if ph < 0.00000:
						ph += 2*np.pi
				
					ph=(np.rint(100*ph)/4).astype(np.int64)

					intcount[ np.rint( (1/targetinc)*(np.linalg.norm(density[i][j].position-mCOM)-lookupdepth[th,ph]) ).astype(np.int64), i ] += 1.0
					radialdensity[ np.rint( (1/targetinc)*(np.linalg.norm(density[i][j].position-mCOM) ) ).astype(np.int64), i ] +=1.0

			for j in range(len(intcount[:,i])):
				if voldist[j]>0:
					intrinsicdensity[j,i] += intcount[j,i]/voldist[j]

print "Done.\n"

#########################################################################################
############################## CURVATURE FUNCTION #######################################
#########################################################################################
if curves==1:
	print "Setting up and compiling curvature function... sit tight.\n"
	# set up derivatives and curvature function required
	gT = [T.grad(pos[0], THETA), T.grad(pos[1], THETA), T.grad(pos[2], THETA)]
	gP = [T.grad(pos[0], PHI), T.grad(pos[1], PHI), T.grad(pos[2], PHI)]
	# SECOND DERIVATIVES
	gTT = [T.grad(gT[0], THETA), T.grad(gT[1], THETA), T.grad(gT[2], THETA)] 
	gPP = [T.grad(gP[0], PHI), T.grad(gP[1], PHI), T.grad(gP[2], PHI)] 
	gTP = [T.grad(gT[0], PHI), T.grad(gT[1], PHI), T.grad(gT[2], PHI)]
	gPT = [T.grad(gP[0], THETA), T.grad(gP[1], THETA), T.grad(gP[2], THETA)]

	E=T.dot(gT,gT) # calculate first fundamental form
	F=T.dot(gT,gP)
	G=T.dot(gP,gP) 

	# calculate second fundamental form
	# calculate normal vector to surface
	n=[ gT[1]*gP[2]-gT[2]*gP[1] , gT[2]*gP[0]-gT[0]*gP[2], gT[0]*gP[1]-gT[1]*gP[0] ] /  T.sqrt( T.dot( [ gT[1]*gP[2]-gT[2]*gP[1] , gT[2]*gP[0]-gT[0]*gP[2], gT[0]*gP[1]-gT[1]*gP[0] ] , [ gT[1]*gP[2]-gT[2]*gP[1] , gT[2]*gP[0]-gT[0]*gP[2], gT[0]*gP[1]-gT[1]*gP[0] ] ) )

	L=T.dot(gTT,n)
	M=T.dot(gTP,n)
	N=T.dot(gPP,n)

	CURVATURE=theano.function([THETA,PHI], (L*N-M**2)/(E*G-F**2) ) 
	print "Done.\n"


print "Setup is complete. Trajectory processing is underway.\n\n"

count = 0
######################################################################################################################
######################################################################################################################
######################################################################################################################
########################################### BEGIN MAIN LOOP ##########################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
for ts in u.trajectory: # loop through trajectory frames here 
	print "Processing snapshot %d " % (ts.frame)

	intrinsiccount=np.zeros((600,len(density)), dtype=np.float32) # arrays for calculating density are reset to {0}
	volatdist=np.zeros((600), dtype=np.float32) # array that stores the instantaneous volume of each spatial interval

	updateCOM(mCOM)
	updateanchors(ancVECS,ancVECsize,mCOM)
	updateSURFACE() # micelle DEPTH values are updated	
	
	if args['XYZsurface']:
		writeSURFACEandCURVATURE(fvissurf,curves) # write micelle surface to xyz file
  
 	if args['density']: 
 		updateVOLESTIMATE(volatdist) # volume estimate is updates for snapshot
		calcdensity(intrinsiccount,volatdist) # calculate density here

	count+=1

##################################
##### Print results to files #####
##################################
if args['density']:

	for i in range(-40,31):
		intdensityout.write("%f " % (targetinc*float(i)) )
		for j in range(len(density)):
			intdensityout.write("%f " % (intrinsicdensity[i,j]/float(count)) ) 
		intdensityout.write("\n")
	intdensityout.write("\n")

	for i in range(0,100):
		raddensityout.write("%f " % (targetinc*float(i)) )
		for j in range(len(density)):
			raddensityout.write("%f " % (radialdensity[i,j]/(float(count)*( (4*np.pi/3)*(i*targetinc+0.5*targetinc)**3 - (4*np.pi/3)*(i*targetinc-0.5*targetinc)**3 ) ) ) ) 
		raddensityout.write("\n")
	intdensityout.write("\n")
	raddensityout.write("\n")

print "Program finished successfully!!!\n\nTime taken to execute analysis was %f\n\n" % (time.time()-start_time)