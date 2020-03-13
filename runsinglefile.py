import membranesimulation as mb
import argparse
import os
import subprocess
from subprocess import call
import pickle
import nanoparticle
import parlammps


parser = argparse.ArgumentParser()

parser.add_argument('-i','--input', default='', type=str, 
                    help='input lammps script file')
parser.add_argument('-o','--out', default='', type=str, 
                    help='output directory')

args = parser.parse_args()

istr = "001010000110100100000010011000100000101100000100000010001010110000010011"
individual = [int(i) for i in istr]

particle = nanoparticle.CoveredNanoParticlePhenome(individual,1,0,11,11)

np = particle.particle

sim = mb.MembraneSimulation(
        args.input.split('/')[-1].split('.')[0],
        np,
        25000,
        0.01,        
        args.out,
        os.path.dirname(args.input),
        "/Users/joel/Projects/optihedron/mem/template/data.template",
        "/Users/joel/Projects/optihedron/mem/template/in.template",
        rAxis=(1,1,0),
        rAmount=3.141        
        )

sim.saveFiles()
scriptPath = os.path.join(sim.filedir,sim.scriptName)
cmd='/Users/joel/Projects/optihedron/lammps/src/lmp_serial < {}'.format(scriptPath)
# process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
call(cmd, shell=True)
# output, error = process.communicate()
outFilePath = os.path.join(sim.outdir,sim.outName)
sim.postProcessOutput(outFilePath)
