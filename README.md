# MRF undersampling optimization
To reproduce the results from the paper *Mitigating undersampling errors in Magnetic Resonance Fingerprinting by sequence optimization* one has to use the following scripts: 
+ main_UEE_phase.py
+ UEE_phase.py
+ Optimisation.py.

The following dictionary files have to be downloaded from the folder *Data structures* in the working directory for the program to function correctly: 
+ Single spiral.npz
+ reference_map.npz, 

where *Single spiral.npz* contains the used spiral and *reference_map.npz* the reference brain scan used for the optimisations. 
The Environment variables can be found in *environment.yml*. To run the code on a cluster with Slurm scheduling the batch file *Batch_UNIX* was added. Insert your email and update the current directory and environment to execute the code successfully (and update for your own system). The optimised sequences used in the paper are stored in the folder *Data structures* with the name *sequences.npy*.

**NOTE:** To run the code on multiple CPUs at the same time (which is highly recommended to achieve reasonable computation times) line 47--49 of *main_UEE_phase.py* should be uncommented while line 54 and 55 should be commented. 

## main_UEE_phase.py
In this script the optimisation variables, the reference map and other model parameters are defined. The numerical optimisation is performed using a sequential (least-square) quadratic programming (SLSQP) tool from **Scipy**. It calls to UEE_phase.py to do the actual undersampling estimation for a certain flip angle pattern on which the optimisation is based. 
Optimized sequence is saved to the working directory as plot and npz file.

## UEE_phase.py
In this script an undersampling error prediction is performed. With the current settings the model predicts the second figure from the second row in *Figure 3* from the paper. This script calls to *Optimisation.py* to calculate the evolution of the magnetisation as well as the derivatives of the magnetisation to the tissue parameters. 

## Optimisation.py
In this script the evolution of the magnetisation as well as the derivatives of the magnetisation to the tissue parameters is calculated in an analytical manner using the Extended Phase Graph (EPG) formalism. 
