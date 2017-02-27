# nanoCISC
Python analysis code to calculate the intrinsic surfaces and density profiles of quasi-spherical nanoparticles.

Written by Daniel Allen.

The nanoCISC ("(Not Always Naturally Obvious) the Continuous Intrinsic Surface and Curvature") software is capable of calculating both intrinsic and radial density profiles, as well as optionally outputting an xyz trajectory of the intrinsic surface and/or local curvature of the user defined and inputted nanoparticle. 

The script requires the NumPy (http://www.numpy.org/), MDAnalysis (http://www.mdanalysis.org/) and Theano (http://deeplearning.net/software/theano/) Python libraries to run.

![Alt text](/sample-images/HR_no-surface.png?raw=true "Nanoparticle")

![Alt text](/sample-images/HR_surface.png?raw=true "Nanoparticle")



# Passing arguments to the script

Arguments are fed into the script on the command line, some arguments are optional whilst others are mandatory. The mandatory arguments are:

```
-prefix "NAME_OF_YOUR_SYSTEM"
-top "topology_file"
-traj "trajectory_file"
-micelle "mask that selects all atoms in nanoparticle"
-anchors "mask that selects all anchor points"
```

The -prefix must specify a string, in quotes, which will set the naming scheme for all of the output files for a particular system or analysis run. The path to the topology and trajectory files must also be in quotes. The -micelle and -anchors arguments should specify a selection for MDAnalysis to make a group of all atoms contained within these groups (examples are given below). Without specifying all of the arguments listed above, the code won't run as this is the basic information required to perform the surface calculation. Note that your topology and trajectory files must be compatible with MDAnalysis (see https://pythonhosted.org/MDAnalysis/documentation\_pages/coordinates/init.html#supported-coordinate-formats). The optional arguments are:

```
-lambda $value_of_lambda
-calcrange $value_of_calcrange
-increment $value_of_spatial_increment
-XYZsurface 1 # yes, output the intrinsic surface
-curves 1 # yes, output the local curvature of the surface in the XYZsurface file
-density "Group_1_label" "Group_1_mask" "Group_2_label" "Group_2_mask" 
		... "Group_N_label" "Group_N_mask"
```

Specify floating point value for free parameter lambda (default is 15.0 as was used in the manuscript). The value of $\lambda$ can be systematically chosen as explained in the manuscript by establishing the typical angle between the vectors $\{\mathbf{s}_i\}$ of nearest neighbor anchor points throughout the trajectory, $\bar{\theta}$, and then using this angle to determine $\lambda$ for a specific system: $\lambda=1/\bar{\theta}^2$. The calcrange can be specified as a floating point value and this is the maximum distance away from micelle centre of mass for which to include particles in the density calculation, the default value is 35.0 which was sufficient for the SDS+TP micelle system. A systematic way to choose this would be to take the average radius of the user defined and inputted nanoparticle, add the size of the required water buffer (e.g 15 \AA) and then multiply this distance by two to arrive at the length of the cubic box for which to calculate the density of particles within. The user may specify the increment for the spatial interval widths as a floating point value, the default value is 1.0 \AA\ and it is not recommended to increase this, but if a higher resolution is required then setting this to 0.5 \AA\ would be a sensible choice. The density flag is used to specify groups of atoms that should be included in the density calculation. These should be specified by stating a string label for a group of atoms, followed by a space and then a string containing a mask selection which includes all the atoms in that group which will be understood by MDAnalysis. Examples of this are shown below. The -XYZsurface and -curves flags are simply switches with boolean logic, set as 1 for true (default is 0/false). Setting -XYZsurface 1 -curves 0 will output the surface without curvature information whereas setting -XYZsurface 1 -curves 1 will output the curvature information as an additional column in the xyz file which can be visualised.

# Basic example of usage
The following example will calculate the intrinsic density of only the oxygen atoms in water molecules, labelled as "Ow" and selected as atom group "type 9"  from the SDS+TP micelle trajectory: 
```
nanoCISC -prefix "SDS-TP" -top "SDS+TP.data" -traj "SDS+TP-trajectory.dcd" -nanoparticle "resname SDS" -anchors "type 8" -density "Ow" "type 9"
```

# Advanced example of usage
The following example will calculate the intrinsic density of the oxygen atoms in water molecules, the hydrogen atoms in water molecules and the sodium counterions from the SDS+TP micelle trajectory. The calcrange, increment and value of lambda have all be explicitly defined. The surface will be written to an xyz file along with the local curvature of the surface:
```
nanoCISC -prefix "SDS-TP" -top "SDS+TP.data" -traj "SDS+TP-trajectory.dcd" -density "Ow" "type 9" "Hw" "type 10" "Na+" "type 1" -nanoparticle "resname SDS" -anchors "type 8" -calcrange 40 -increment 0.5 -lambda 10.0 -XYZsurface 1 -curves 1
```
