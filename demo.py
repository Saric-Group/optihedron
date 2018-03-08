"""
This constructs a single nanoparticle with 20 ligands and outputs a data file containing data for
that nanoparticle and a membrane, and a LAMMPS script to read that data and run a demo simulation.
If run as a python script it will run the generated LAMMPS script and produce plotable output in a
'demo' folder.
"""
    
from nanoparticle import NanoParticle, Ligand
from membranesimulation import MembraneSimulation

import parlammps

import math
import os
import argparse

parser = argparse.ArgumentParser(description='Creates and runs a demo simulation.')

#MPI Options
parser.add_argument('-mpi','--mpi', action='store_true', help='option to run in parallel')
parser.add_argument('-np','--nodes', default=4, type=int, help='number of cores used per mpi process')
parser.add_argument('-tm','--timeout', default=1800, type=int, help='mpirun timeout')

parser.add_argument('-c','--clean', action='store_true', help='option to remove data and script files when done')

args = parser.parse_args()

# nanoparticle parameters don't change a thing because they are hardcoded in the data & script templates !!?!
np = NanoParticle()
for i in range(4):
    phi = i*(math.pi/2)
    for j in range(5):
        theta = math.pi/10 + j*(math.pi/5)
        np.addLigand(Ligand(rad=np.sig, polAng=theta, aziAng=phi, mass=1.0, eps=10.0, sig=1.0))

wd = os.path.dirname(os.path.realpath(__file__))
        
simulation = MembraneSimulation("demo", np, 10000, 0.01, os.path.join(wd,'demo'), os.path.join(wd,'demo'),
                                os.path.join(wd,'mem/template/data.template'),
                                os.path.join(wd,'mem/template/in.template'),
                                corepos_x=0.0, corepos_y=0.0, corepos_z=7.0)

simulation.saveFiles()
script = os.path.join(simulation.filedir, simulation.scriptName)
    
if (args.mpi):
    parlammps.runSim(script, args.nodes, args.timeout)
else:
    parlammps.runSimSerial(script)

if (args.clean):
    simulation.deleteFiles()
