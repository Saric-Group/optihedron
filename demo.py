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

# enable options to run parallel, give number of threads to use, decide whether to generate/keep
# the script/data files, or just run via PyLammps, or delete them when simulation is done etc.
if __name__ == '__main__':
    
    parlammps.runSimSerial(os.path.join(simulation.filedir, simulation.scriptName))
    
    # maybe add PyLammps code that runs the demo without using the script, to show the alternative...
