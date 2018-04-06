import os
import errno
import array
import random
import math
import csv
import argparse
import time
import itertools
import operator
import numpy
import joblib
import sys
import subprocess
import parlammps
import pathos
from pathos import pools
import traceback

from operator import itemgetter

# from joblib import Parallel, delayed, parallel_backend
# from distributed.joblib import DaskDistributedBackend
from threading import Timer

from deap import base
from deap import creator
from deap import tools

from ga import networks
from ga import algorithms
from ga import phenome
from ga import networkedgeneticalgorithm as nga
from ga import grayencoder as ge

from tools import misctools
from tools import listtools
from tools import qtools
from tools import vectools

from db import databaseconnection

from lammps import lammps

from nanoparticle import NanoParticlePhenome
from nanoparticle import CoveredNanoParticlePhenome
from membranesimulation import MembraneSimulation

parser = argparse.ArgumentParser(description='')

#Runtime Options 

parser.add_argument('-v','--verbose', default=False, action='store_true',
                    help='option to run in verbose mode')

#Genetic Algorithm Options

parser.add_argument('-n','--ngen',type=int,
                    help='number of generations', required=True)
parser.add_argument('-d','--demes', type=int,
                    help='number of demes to run over', required=True)
parser.add_argument('-p','--pop', type=int,
                    help='population of each deme', required=True)
parser.add_argument('-gs','--genomesize', type=int,
                    help='number of bits in the genome', default=648)
parser.add_argument('-f','--migfreq', type=int, default=1,
                    help='number of generations between migrations')
parser.add_argument('-c','--cxpb', default=0.5,  type=float,
                    help='independant probability of crossover')
parser.add_argument('-m','--mutpb', default=0.2, type=float,
                    help='independant probability of mutation')
parser.add_argument('-mg','--migrations', default=1, type=int,
                    help='number of migrations to do each time')
parser.add_argument('-mpb','--mindpb', default=0.05, type=float,
                    help='independant probability of a bit flip')
parser.add_argument('-t','--tournsize', default=3, type=int,
                    help='size of selection tournaments')
parser.add_argument('-g','--graph', default='islands', 
                    choices=['singlet','islands','star','megastar'],
                    help='type of network to use')
parser.add_argument('-a', '--algorithm', default='eaSimple',
                    choices=['eaSimple'])

#Model Options

parser.add_argument('-s','--seed', default=int(time.time()), type=int,
                    help='seed for the RNG')
parser.add_argument('-hof','--hofsize', default=5, type=int,
                    help='hall of fame size')
parser.add_argument('-expr','--exprplaces', default=1, type=int,
                    help='number of bits for ligand expression')
parser.add_argument('-eps','--epsplaces', default=8, type=int,
                    help='number of bits for epsilon')
parser.add_argument('-polang','--polangplaces', default=8, type=int,
                    help='number of bits for polar angle')
parser.add_argument('-aziang', '--aziangplaces', default=8, type=int,
                    help='number of bits for azimuthal angle')
parser.add_argument('-epmn','--epsmin', default=0, type=float,
                    help='minimum value for epsilon')
parser.add_argument('-epmx','--epsmax', default=10, type=float,
                    help='maximum value for epsilon')
parser.add_argument('-r','--runtime', default=50000, type=int,
                    help='lammps timesteps')
parser.add_argument('-ts','--timestep', default=0.01, type=float,
                    help='lammps timestep size')
parser.add_argument('-rs','--repeats', default=4, type=int,
                    help='number of repeat tests for each individual')
parser.add_argument('-pw','--penaltyweight', default=1.0, type=float,
                    help='weighting of the ligand affinity penalty')
parser.add_argument('-pp','--partialpacking', action='store_true',
                    help='option to run the algorithm with partially packed sphere. In this mode, the azimuthal and polar angles will be controlled by the genome')

#Concurrency Options

parser.add_argument('-mpi','--mpi', action='store_true',
                    help='option to run in parallel')
parser.add_argument('-q','--qsub', action='store_true',
                    help='option to qsub to cluster')
parser.add_argument('-w','--workers', default=10, type=int,
                    help='number of workers in the mapping pool')
parser.add_argument('-np','--nodes', default=4, type=int,
                    help='number of cores used per mpi process')
parser.add_argument('-tm','--timeout', default=1800, type=int,
                    help='mpirun timeout')

#Data Options

parser.add_argument('-sr','--saveresults', action='store_true',
                    help='option to save results to db')
parser.add_argument('-ko','--keepoutput', action='store_true',
                    help='option to keep all output files the simulation generates')
parser.add_argument('-ki','--keepinput', action='store_true',
                    help='option to keep all input files the simulation generates')
parser.add_argument('-kb','--keepbest', action='store_true',
                    help='option to store each generations best individual')
parser.add_argument('-wd','--wdir', default=os.path.dirname(os.path.realpath(__file__)),
                    help='option to set the working directory of the program')

args = parser.parse_args()

NSPOKES = 2
NISLES = args.demes
ISLESIZE = args.pop
CXPB=args.cxpb
MUTPB=args.mutpb
NGEN, FREQ = args.ngen, args.migfreq
VERBOSE=args.verbose
SEED = args.seed
TSIZE = args.tournsize
MINPDB = args.mindpb
GENOMESIZE = args.genomesize
MIGR = args.migrations
HOFSIZE = args.hofsize
EXPRPLACES = args.exprplaces
EPSPLACES = args.epsplaces
POLANGPLACES = args.polangplaces
AZIANGPLACES = args.aziangplaces
EPSMAX = args.epsmax
EPSMIN = args.epsmin
RUNTIME = args.runtime
TIMESTEP = args.timestep
PARTIAL = args.partialpacking
GENESIZE = (EXPRPLACES+EPSPLACES+POLANGPLACES+AZIANGPLACES) if PARTIAL else (EXPRPLACES+EPSPLACES)
GENES = math.floor(GENOMESIZE/GENESIZE)
QSUB = args.qsub
WORKERS = args.workers
REPEATS = args.repeats
KEEPINPUT = args.keepinput
KEEPOUTPUT = args.keepoutput
KEEPBEST = args.keepbest
PENALTYWEIGHT = args.penaltyweight

WDIR = args.wdir
PDIR = os.path.dirname(os.path.realpath(__file__))

OUTDIR = os.path.join(WDIR,'out')
RUNDIR = os.path.join(WDIR,'run')
HOFDIR = os.path.join(WDIR,'hof')
DBDIR = os.path.join(WDIR,'db')
TEMPLATEDIR = os.path.join(PDIR,'mem/template')
TEMPLATEDATAPATH = os.path.join(TEMPLATEDIR,'data.template')
TEMPLATEINPUTPATH = os.path.join(TEMPLATEDIR,'in.template')
DBPATH = os.path.join(DBDIR,'datastore.db')


MPI = args.mpi
NP = args.nodes
TIMEOUT = args.timeout

SAVERESULTS = args.saveresults

def kill(p):
    try:
        print("killing "+str(p))
        p.kill()
    except OSError:
        pass # ignore

def exit(signal, frame):
        os._exit(1)

def saveMetrics(lis,filename='metrics.csv'):
    with open(os.path.join(WDIR,filename),'wb') as out:
        csv_out=csv.DictWriter(out,lis[-1].keys())
        csv_out.writeheader()
        for row in lis:
            csv_out.writerow(row)

def evaluateNPWrapping(np,outFilename,runtime):    
    minFit = 1E-8
    outHeaderSize = 9
    outData = {}

    nActiveLigands = 0
    npTotalEps = 0.0

    for l in np.ligands:
        if l.eps > 0.0:
            nActiveLigands += 1
        npTotalEps += l.eps

    if(not os.path.exists(outFilename)):                                
            return minFit,

    with open(outFilename, 'r+') as f:
        lines = f.readlines()
        ts = 0
        steps = []
        hPos = 0        
        for i in range(len(lines)):
            if str('ITEM: TIMESTEP') in lines[i]:
                hPos = i
                ts = int(lines[i+1])
                steps.append(ts)
                outData[ts]=[]                
            if(i-hPos>outHeaderSize):
                outData[ts].append(lines[i].replace("\n","").replace(" ",","))

    if len(outData[ts])<50:        
        return minFit, 

    stepData = []

    for s in steps:   
        outVectors = {}
        for line in outData[ts]:
            slist = line.split(",")[1:]
            sId = line.split(",")[0]
            if(len(slist)<3):            
                return minFit,
            if not int(slist[0]) in outVectors:
                outVectors[int(slist[0])] = []
            outVectors[int(slist[0])].append({'id':sId,'x':float(slist[1]),'y':float(slist[2]), 'z':float(slist[3]), 'c':int(slist[4])})

        cStep = []
        mStep = []
        boxsize = 20
        for key, value in outVectors.iteritems():
            budded = False
            for v in value:
                cIds = [c['id'] for c in cStep]
                if not v['c'] in cIds:
                    cStep.append({'id':v['c'],'size':1})
                else:
                    cId = 0
                    cCount = 0
                    for cI in cIds:
                        if cI == v['c']:
                            cId = cCount
                        cCount += 1
                    cStep[cId]['size'] += 1

            if key == 2:
                for v in value:
                    inrange = 0
                    fmag = 0
                    for v2 in outVectors[1]:
                        xd = v['x']-v2['x']
                        yd = v['y']-v2['y']
                        zd = v['z']-v2['z']
                        #squared magnitude of the difference
                        m = xd*xd+yd*yd+zd*zd                                          
                        if(m<25.0):
                            mStep.append(v2['id'])

            nLargeClusters = 0
            for v in sorted(cStep, key=itemgetter('size')):
                if v['size'] > 100:
                    nLargeClusters += 1
            budded = nLargeClusters > 1
                                       
            stepData.append({'timestep':s,'clusters':cStep,'magnitudes':mStep,'cNum':len(cStep),'mNum':len(mStep), 'budded': budded})


    msum = stepData[-1]['mNum']

    if(msum == 0):        
        return minFit,

    reward = 400 if stepData[-1]['budded'] else 0

    penalty = PENALTYWEIGHT*(1.0-(float(npTotalEps)/(float(EPSMAX)*float(nActiveLigands))))*100 if stepData[-1]['budded'] and float(EPSMAX)*float(nActiveLigands) > 0.0 else 0.0

    return msum + reward + penalty,

def runCmd(cmd,timeout):
    try:
        with open('subprocess.log', 'w+') as f:
            proc = subprocess.Popen(
                cmd,
                shell=True,
                stdout=f, 
                stderr=f
                )
            t = Timer(timeout, kill, [proc])
            t.start()
            proc.wait()
            t.cancel()
    except Exception as e: 
        print(cmd)
        print(e)

def makeXYZTriplet(individual,xMax,xMin,yMax,yMin,zMax,zMin):    
    thirds = listtools.subdivide(individual,int(len(individual)*(1.0/3.0)))
    a = ge.read(thirds[0])
    b = ge.read(thirds[1])
    c = ge.read(thirds[2])
    maxA = ge.max(thirds[0])
    maxB = ge.max(thirds[0])
    maxC = ge.max(thirds[0])
    xRange = xMax - xMin
    yRange = yMax - yMin
    zRange = zMax - zMin
    x=float((a/float(maxA))*xRange + xMin)
    y=float((b/float(maxB))*yRange + yMin)
    z=float((c/float(maxC))*zRange + zMin)
    return x,y,z    

def evaluatePyLammps(individual):

    return 1,

def runSim(path):    
    return parlammps.runSim(path,NP,TIMEOUT) if MPI else parlammps.runSimSerial(path)

def evaluateParticleInstance(np,simName):
    
    sim = MembraneSimulation(
        'sim_'+simName,
        np,
        RUNTIME,
        TIMESTEP,        
        OUTDIR,
        RUNDIR,
        TEMPLATEDATAPATH,
        TEMPLATEINPUTPATH,
        rAxis=vectools.randomUnitVector(),
        rAmount=random.uniform(0.3141,3.141)        
        )
    sim.saveFiles()
    scriptPath=os.path.join(sim.filedir,sim.scriptName)
    outFilePath = os.path.join(sim.outdir,sim.outName)
    time.sleep(1)

    if(QSUB):
        try:
            pbs = parlammps.createPbs(scriptPath,WDIR,NP,simName,sim.filedir,MPI)
            job = subprocess.Popen(["qsub", pbs],stdout=subprocess.PIPE)
            job.wait()
            out = job.communicate()[0]
            while(qtools.hasRQJob(out)):
                time.sleep(1)
            os.remove(pbs)
            
        except Exception as err:
            traceback.print_exc()
            print('error in qsub of {}, file: '.format(simName,pbs))
    else:
        runSim(scriptPath)
    
    f = 1E-8,
    f = evaluateNPWrapping(np,outFilePath,RUNTIME)

    print('{} fitness: {}'.format(simName, f))
    if not KEEPINPUT:
        sim.deleteFiles()
    if KEEPOUTPUT:
        sim.postProcessOutput(outFilePath)
    else:
        os.remove(outFilePath)
    return f


def evaluate(individual):
    phenome = CoveredNanoParticlePhenome(individual,EXPRPLACES,EPSPLACES,EPSMIN,EPSMAX) if not PARTIAL else NanoParticlePhenome(individual,EXPRPLACES,EPSPLACES,POLANGPLACES,AZIANGPLACES,EPSMIN,EPSMAX)
    
    np = phenome.particle
    simName = misctools.randomStr(10)
    fitnesses = []

    for i in range(REPEATS):
        fitnesses.append(evaluateParticleInstance(np,simName+"_"+str(i)))

    fsum = 0
    for fit in fitnesses:
        fsum+=fit[0]

    fmem = float(fsum)/float(REPEATS)

    #flig = len(np.ligands) 
    #feps = sum([ligand.eps for ligand in np.ligands])

    f = fmem

    return f,

def sel(pop,k):
    return tools.selTournament(pop,k,TSIZE)

def mut(individual):
    return tools.mutFlipBit(individual,MINPDB)

def algorithmEaSimple(pop,toolbox,stats,hof):
    return algorithms.eaSimple(pop,toolbox=toolbox,
        cxpb=CXPB, mutpb=MUTPB, ngen=FREQ,verbose=VERBOSE,stats=stats,halloffame=hof)

def beforeMigration(ga):
    misctools.removeByPattern(WDIR,"subhedra")

    if SAVERESULTS:
        ind = 0
        for isle in ga.islands:
            for individual in isle:
                np = CoveredNanoParticlePhenome(individual,EXPRPLACES,EPSPLACES,EPSMIN,EPSMAX) if not PARTIAL else NanoParticlePhenome(individual,EXPRPLACES,EPSPLACES,POLANGPLACES,AZIANGPLACES,EPSMIN,EPSMAX)
                ga.dbconn.saveIndividual(ga.gen, ind, individual.fitness.values[-1], individual, np)
                ind += 1
        ga.dbconn.commit()
    if KEEPBEST:
        saveBest(ga.hof,ga.gen)

    return

def afterMigration(ga):
    outFile = ""
    isleNum = 0
    i = 0
    for isle in ga.islands:
        isleNum += 1
        points = [makeXYZTriplet(p,2,-2,2,-2,2,-2) for p in isle]
        fit = [p.fitness.values[-1] for p in isle]
        for p in points:
            i+=1
            outFile += str(i)+","+str(p[0])+","+str(p[1])+","+str(p[2])+"\n"
    with open(os.path.join(WDIR,'coords.csv'), 'a') as file_:    
        file_.write(outFile)
    return

def saveBest(hof,gen):

    if len(hof) < 1:
        return
    ind = hof[0]
    phenome = CoveredNanoParticlePhenome(ind,EXPRPLACES,EPSPLACES,EPSMIN,EPSMAX) if not PARTIAL else NanoParticlePhenome(ind,EXPRPLACES,EPSPLACES,POLANGPLACES,AZIANGPLACES,EPSMIN,EPSMAX)
    np = phenome.particle
    sim = MembraneSimulation(
        'gen_'+str(gen)+"_best",
        np,
        RUNTIME,
        TIMESTEP,            
        OUTDIR,
        HOFDIR,            
        TEMPLATEDATAPATH,
        TEMPLATEINPUTPATH 
        )
    hofScriptPath = os.path.join(sim.filedir,sim.scriptName)
    sim.saveFiles()
    runSim(hofScriptPath)
    outFilePath = os.path.join(sim.outdir,sim.outName)
    sim.postProcessOutput(outFilePath)

def saveHOF(hof):
    i = 1
    for ind in hof:
        phenome = CoveredNanoParticlePhenome(ind,EXPRPLACES,EPSPLACES,EPSMIN,EPSMAX) if not PARTIAL else NanoParticlePhenome(ind,EXPRPLACES,EPSPLACES,POLANGPLACES,AZIANGPLACES,EPSMIN,EPSMAX)
        np = phenome.particle
        sim = MembraneSimulation(
            'hof_'+str(i),
            np,
            RUNTIME,
            TIMESTEP,            
            OUTDIR,
            HOFDIR,            
            TEMPLATEDATAPATH,
            TEMPLATEINPUTPATH 
            )
        hofScriptPath = os.path.join(sim.filedir,sim.scriptName)
        sim.saveFiles()
        runSim(hofScriptPath)
        outFilePath = os.path.join(sim.outdir,sim.outName)
        sim.postProcessOutput(outFilePath)


def geneWiseTwoPoint(ind1,ind2):
    size = len(ind1)
    cxpoint1 = random.randint(1, GENES)*GENESIZE
    cxpoint2 = random.randint(1, GENES-1)*GENESIZE
    if cxpoint2 >= cxpoint1:
        cxpoint2 += GENESIZE
    else: # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()
        
    return ind1, ind2

def mkdirs(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise        

def main():

    try:
        if not os.path.isdir(WDIR):
            mkdirs(WDIR)
        if not os.path.isdir(OUTDIR):
            mkdirs(OUTDIR)
        if not os.path.isdir(HOFDIR):
            mkdirs(HOFDIR)
        if not os.path.isdir(DBDIR):
            mkdirs(DBDIR)
        if not os.path.isdir(RUNDIR):
            mkdirs(RUNDIR)
    except Exception as e:
        print "error creating working environment"
        raise e
        
    if args.graph == 'singlet':
       network = networks.createSinglets(NISLES)
    elif args.graph == 'islands':
       network = networks.createIslands(NISLES)
    elif args.graph == 'star':
       network = networks.createStar(NISLES-1)
    elif args.graph == 'megastar':
       network = networks.createMegaStar(NSPOKES,int(math.ceil((NISLES-3)*0.25)),int(math.floor((NISLES-3)*0.25)))
    else:
       raw_input('malformed network option, continue with islands? (Enter)')

    if args.algorithm == 'eaSimple':
        algorithm = algorithmEaSimple
    else:
        raw_input('malformed algorithm option, continuing with eaSimple.. (Enter)')
        algorithm = algorithmEaSimple

    random.seed(args.seed)

    if SAVERESULTS:
       dbconn = databaseconnection.DatabaseConnection(DBPATH)
       dbconn.saveSession()
       dbconn.commit()
    else:
       dbconn = None

    ga = nga.NetworkedGeneticAlgorithm(
       genomeSize = GENOMESIZE,
       islePop = ISLESIZE,
       hofSize = HOFSIZE,
       evaluate = evaluate,
       sel = sel,
       net = network,
       subroutine = algorithm,
       mut = mut,
       mapping = pools.ProcessPool(WORKERS).map,
       beforeMigration = beforeMigration,
       afterMigration = afterMigration,
       verbose = VERBOSE,
       mate = geneWiseTwoPoint,
       dbconn = dbconn)

    results = ga.run(NGEN,FREQ,MIGR)



   #print(results[0][0][0])
    
    saveMetrics(results[-2])
    saveHOF(results[1])

    if SAVERESULTS:
       dbconn.saveMetrics(results[-2])
       dbconn.saveGenealogy(results[-1].genealogy_tree, results[-1].genealogy_history)
       dbconn.commit()
       dbconn.close()         


if __name__ == "__main__":
    main()