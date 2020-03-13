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
import sys
import subprocess
import parlammps
import pathos
import sys
from pathos import pools
import traceback
import pickle

from operator import itemgetter

# from joblib import Parallel, delayed, parallel_backend
# from distributed.joblib import DaskDistributedBackend
from threading import Timer

from deap import base
from deap import creator
from deap import tools

from ga import networks
from ga import algorithms
from ga import operators
from ga import phenome
from ga import networkedgeneticalgorithm as nga
from ga import grayencoder as ge

from tools import misctools
from tools import listtools
from tools import qtools
from tools import vectools
from tools import analysistools as atools

from db import databaseconnection as dao

from lammps import lammps

from nanoparticle import NanoParticlePhenome
from nanoparticle import CoveredNanoParticlePhenome
from membranesimulation import MembraneSimulation

import json

import networkx as nx

parser = argparse.ArgumentParser(description='')

#Runtime Options 

parser.add_argument('-v','--verbose', default=False, action='store_true',
                    help='option to run in verbose mode')

#Genetic Algorithm Options

parser.add_argument('-n','--ngen',type=int,
                    help='number of generations', default=1)
parser.add_argument('-d','--demes', type=int,
                    help='number of demes to run over', default=1)
parser.add_argument('-p','--pop', type=int,
                    help='population of each deme', default=1)
parser.add_argument('-gs','--genomesize', type=int,
                    help='number of genes in the genome (overwritten for fixed angle)', default=40)
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
parser.add_argument('-mo', '--mutation', default='defaultMut',
                    choices=['defaultMut', 'fixedActivationMut'])
parser.add_argument('-xo', '--mate', default='defaultGeneWiseTwoPoint',
                    choices=['defaultGeneWiseTwoPoint', 'fixedActivationGeneWiseTwoPoint'])
parser.add_argument('-sg','--startinggen',default=0, type=int,
                    help='starting generation')

parser.add_argument('-ff','--fitnessfunction',default="budding",
                    choices=['budding', 'diameter','smallworld'])


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
parser.add_argument('-epmx','--epsmax', default=15, type=float,
                    help='maximum value for epsilon')
parser.add_argument('-r','--runtime', default=25000, type=int,
                    help='lammps timesteps')
parser.add_argument('-ts','--timestep', default=0.01, type=float,
                    help='lammps timestep size')
parser.add_argument('-rs','--repeats', default=4, type=int,
                    help='number of repeat tests for each individual')
parser.add_argument('-br', '--buddingreward',default=400.0, type=float,
                    help='reward for successful budding in')
parser.add_argument('-pw','--penaltyweight', default=10.0, type=float,
                    help='weighting of the ligand affinity penalty')
parser.add_argument('-tw','--timeweight', default=25.0, type=float,
                    help='weighting of the budding time reward')
parser.add_argument('-lw','--ligandweight', default=10.0, type=float,
                    help='weighting of the target ligand reward')
parser.add_argument('-tl','--targetligands', default=-1, type=int,
                    help='ideal number of ligands')
parser.add_argument('-fl','--fixedligands',default=-1, type=int,
                    help='fixed number of ligands to use for individual')
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
parser.add_argument('-kw','--keepworst', action='store_true',
                    help='option to store each generations worst individual')
parser.add_argument('-wd','--wdir', default=os.path.dirname(os.path.realpath(__file__)),
                    help='option to set the working directory of the program')
parser.add_argument('-i','--input', default=None, type=str, 
                    help='set the input json file')
parser.add_argument('-db','--database', default=None, type=str, 
                    help='set the input db')
parser.add_argument('-dbs','--dbsession', default=-1, type=int, 
                    help='set the session to load from the db')
parser.add_argument('-z','--zoo', default=None, type=str, 
                    help='set the input zoo, does not run a GA in this mode')



args = parser.parse_args()

FILE = args.input
DB = args.database
DBSESSION = args.dbsession
LOADFROMFILE = False
LOADFROMDB = False

WDIR = args.wdir
PDIR = os.path.dirname(os.path.realpath(__file__))

OUTDIR = os.path.join(WDIR,'out')
RUNDIR = os.path.join(WDIR,'run')
HOFDIR = os.path.join(WDIR,'hof')
DBDIR = os.path.join(WDIR,'db')

if not os.path.isdir(OUTDIR):
    os.makedirs(OUTDIR)

if not os.path.isdir(RUNDIR):
    os.makedirs(RUNDIR)

if not os.path.isdir(HOFDIR):
    os.makedirs(HOFDIR)

if not os.path.isdir(DBDIR):
    os.makedirs(DBDIR)


TEMPLATEDIR = os.path.join(PDIR,'mem/template')
TEMPLATEDATAPATH = os.path.join(TEMPLATEDIR,'data.template')
TEMPLATEINPUTPATH = os.path.join(TEMPLATEDIR,'in.template')
DBPATH = os.path.join(DBDIR,'datastore.db')

QSUB = args.qsub
WORKERS = args.workers

MPI = args.mpi
NP = args.nodes
TIMEOUT = args.timeout

SAVERESULTS = args.saveresults

KEEPINPUT = args.keepinput
KEEPOUTPUT = args.keepoutput
KEEPBEST = args.keepbest
KEEPWORST = args.keepworst

ZOO = args.zoo

runArgs = args


if DB != None:
    try:
        conn = dao.DatabaseConnection(DB)
        print('loading database file ' + str(DB))
        if DBSESSION == -1:
            print('loading most recent db session')
        else:
            print('loading session ' + str(DBSESSION))
        initSession = conn.getSession(DBSESSION) if DBSESSION != -1 else conn.getLastSession()

        if initSession != None:
            lastGen = initSession.getLastGeneration()
            runArgs = initSession.argPickle
            sGen = lastGen.genNumber+1
            setattr(runArgs,"startinggen",sGen)
            if args.ngen > runArgs.ngen:
                setattr(runArgs,"ngen",args.ngen)
            initpop = []
            if initSession.demes != None and len(initSession.demes) > 0:
                print('loading deme lists')
                for d in initSession.demes:
                    initpop.append([])
                    for ind in d.individuals:
                        if ind.gen_id == lastGen.pID:
                            initpop[-1].append(numpy.array(ind.genomePickle).tolist())
            else:
                nDemes = runArgs.demes
                print('no deme lists, splitting population into ' + str(nDemes) + ' new demes')
                for d in range(nDemes):
                    initpop.append([])
                    for ind in range(d*runArgs.pop,d*runArgs.pop+runArgs.pop):
                        initpop[-1].append(numpy.array(lastGen.individuals[ind].genomePickle).tolist())
            initParams = {'init_pop':initpop}
            initFileName = 'db/init.json'
            with open(initFileName, 'w') as initFile:
                json.dump(initParams, initFile)
            FILE = initFileName
        else:
            print("database contained no valid sessions")
        conn.close()
    except Exception as e: 
        print(e)
        
#sanitize input
if FILE != None:
    try:
        with open(FILE, "r") as pop_file:
            contents = json.load(pop_file)
        LOADFROMFILE = "init_pop" in contents
        for arg in vars(args):
            if str(arg) in contents:
                print('overwriting ' + str(arg) + ' with value from file: '+ str(contents[str(arg)]))
                setattr(args,arg,contents[str(arg)])
        if LOADFROMFILE:
            print('overwriting deme and population size from init_pop')
            setattr(runArgs,"demes",len(contents['init_pop']))
            setattr(runArgs,"pop",len(contents['init_pop'][0]))

    except:
        print("error loading json")

if args.epsplaces == 0:
    if args.epsmax != args.epsmin:
        print('ligand eps not encoded in genome but requested max and min ligand eps differ')
        print('overwriting max ligand eps ({}) with min ligand eps ({})'.format(args.epsmax, args.epsmin))
        args.epsmax = args.epsmin



NSPOKES = 2
NISLES = runArgs.demes
ISLESIZE = runArgs.pop
CXPB = runArgs.cxpb
MUTPB = runArgs.mutpb
NGEN = runArgs.ngen
FREQ = runArgs.migfreq
VERBOSE=runArgs.verbose
SEED = runArgs.seed
TSIZE = runArgs.tournsize
MINPDB = runArgs.mindpb
PARTIAL = runArgs.partialpacking
GENES = runArgs.genomesize if PARTIAL else 72
MIGR = runArgs.migrations
HOFSIZE = runArgs.hofsize
EXPRPLACES = runArgs.exprplaces
EPSPLACES = runArgs.epsplaces
POLANGPLACES = runArgs.polangplaces
AZIANGPLACES = runArgs.aziangplaces
EPSMAX = runArgs.epsmax
EPSMIN = runArgs.epsmin
RUNTIME = runArgs.runtime
TIMESTEP = runArgs.timestep
GENESIZE = (EXPRPLACES+EPSPLACES+POLANGPLACES+AZIANGPLACES) if PARTIAL else (EXPRPLACES+EPSPLACES)
GENOMESIZE = GENES*GENESIZE
REPEATS = runArgs.repeats
ZOO = runArgs.zoo

PENALTYWEIGHT = runArgs.penaltyweight
TARGETWEIGHT = runArgs.ligandweight
TARGETLIGANDS = runArgs.targetligands
BUDDINGREWARD = runArgs.buddingreward
TIMEWEIGHT = runArgs.timeweight
STARTINGGEN = runArgs.startinggen

FIXEDLIGANDS = runArgs.fixedligands

PARTICLES = []

FITNESSFUNCTION = runArgs.fitnessfunction

#god what a mess

if runArgs != args:
    setattr(args,"demes",NISLES)
    setattr(args,"pop",ISLESIZE)
    setattr(args,"cxpb",CXPB)
    setattr(args,"mutpb",MUTPB)
    setattr(args,"ngen",NGEN)
    setattr(args,"migfreq",FREQ)
    setattr(args,"verbose",VERBOSE)
    setattr(args,"seed",SEED)
    setattr(args,"tournsize",TSIZE)
    setattr(args,"mindpb",MINPDB)
    setattr(args,"partialpacking",PARTIAL)
    setattr(args,"genes",GENES)
    setattr(args,"migrations",MIGR)
    setattr(args,"hofsize",HOFSIZE)
    setattr(args,"exprplaces",EXPRPLACES)
    setattr(args,"epsplaces",EPSPLACES)
    setattr(args,"polangplaces",POLANGPLACES)
    setattr(args,"aziangplaces",AZIANGPLACES)
    setattr(args,"epsmin",EPSMIN)
    setattr(args,"epsmax",EPSMAX)
    setattr(args,"runtime",RUNTIME)
    setattr(args,"timestep",TIMESTEP)
    setattr(args,"repeats",REPEATS)
    setattr(args,"penaltyweight",PENALTYWEIGHT)
    setattr(args,"buddingreward",BUDDINGREWARD)
    setattr(args,"startinggen",STARTINGGEN)
    setattr(args,"timeweight",TIMEWEIGHT)
    setattr(args,"ligandweight",TARGETWEIGHT)
    setattr(args,"targetligands",TARGETLIGANDS)
    setattr(args,"fixedligands",FIXEDLIGANDS)
    setattr(args,"zoo",ZOO)
    setattr(args,"fitnessfunction",FITNESSFUNCTION)


def kill(p):
    try:
        print("killing "+str(p))
        p.kill()
    except OSError:
        pass # ignore

def exit(signal, frame):
        os._exit(1)

def saveMetrics(lis,filename='metrics.csv'):
    if len(lis)>0:
        with open(os.path.join(WDIR,filename),'wb') as out:
            csv_out=csv.DictWriter(out,lis[-1].keys())
            csv_out.writeheader()
            for row in lis:
                csv_out.writerow(row)

def evaluateNPWrapping(np,outFilename,runtime):    
    minFit = 1E-8
    noBud = False
    outHeaderSize = 9
    outData = {}

    nActiveLigands = 0
    npTotalEps = 0.0

    ligMap = []
    builtLigMap = False

    for l in np.ligands:
        nActiveLigands += 1
        npTotalEps += l.eps


    if(not os.path.exists(outFilename)):                                
            return minFit, noBud

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
        return minFit, noBud 

    stepData = []
    dbSteps = []

    for s in steps:   
        outVectors = {}
        for line in outData[s]:
            slist = line.split(",")[1:]
            sId = line.split(",")[0]
            if(len(slist)<3):            
                return minFit, noBud
            if not int(slist[0]) in outVectors:
                outVectors[int(slist[0])] = []
            outVectors[int(slist[0])].append({'id':sId,'x':float(slist[1]),'y':float(slist[2]), 'z':float(slist[3]), 'c':int(slist[4])})

        cStep = []
        mStep = []
        lStep = []
        ligandsInContact = 0
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
                    contact = False
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
                            

        if not builtLigMap:
            for key, value in outVectors.iteritems():
                if key > 2:
                    for v in value:
                        ligMap.append(v['id'])
            builtLigMap = True

        for key, value in outVectors.iteritems():
            if key > 2:
                for v in value:
                    for c in cStep:
                        if c['id'] == v['c']:
                            if c['size'] > 5:
                                ligandsInContact+=1
                                lStep.append(ligMap.index(v['id']))

        nLargeClusters = 0
        for v in sorted(cStep, key=itemgetter('size')):
            if v['size'] > 250:
                nLargeClusters += 1
        percentCoverage = 0
        if len(outVectors[2])>0:
            percentCoverage = float(ligandsInContact)/float(len(outVectors[2]))
        budded = nLargeClusters > 1                        
        stepData.append({'timestep':s,'clusters':cStep,'magnitudes':mStep,'walk':list(set(lStep)),'cNum':len(cStep),'mNum':len(mStep), 'budded': budded, 'coverage':percentCoverage,'contact': ligandsInContact})
        dbSteps.append({'timestep':s,'cNum':len(cStep),'mNum':len(mStep), 'budded': budded, 'coverage':percentCoverage, 'contact': ligandsInContact})
    msum = stepData[-1]['mNum']
    lstep = stepData[-1]['timestep']

    if(msum == 0):        
        return minFit, noBud

    budTime = -1
    for step in stepData:
        if step['budded'] == True:
            budTime = step['timestep']
            break

    
    #reward = msum

    # penalty = PENALTYWEIGHT*(1.0-(float(npTotalEps)/(float(EPSMAX)*float(GENES))))*100 if float(EPSMAX)*float(nActiveLigands) > 0.0 else 0.0

    # reward = (float(BUDDINGREWARD) + float(penalty)) if stepData[-1]['budded'] else float(msum)
    if stepData[-1]['budded'] and budTime != 0 :
        reward = float(msum) + float(float(TIMEWEIGHT)*(float(lstep)/float(budTime))) + float(BUDDINGREWARD)
    else:
        reward = float(msum)

    return reward, stepData[-1]['budded'], budTime, dbSteps, stepData

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

def evaluateParticleInstance(np,simName,rVec=vectools.randomUnitVector(),rAm=random.uniform(0.3141,3.141)):
    
    sim = MembraneSimulation(
        'sim_'+simName,
        np,
        RUNTIME,
        TIMESTEP,        
        OUTDIR,
        RUNDIR,
        TEMPLATEDATAPATH,
        TEMPLATEINPUTPATH,
        rAxis=rVec,
        rAmount=rAm        
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
            if os.path.exists(pbs):
                os.remove(pbs)
                print "removed file: " + str(pbs)
            
        except Exception as err:
            traceback.print_exc()
            print('error in qsub of {}, file: '.format(simName,pbs))
    else:
        runSim(scriptPath)

    time.sleep(1)
    f = 1E-8
    b = False
    bt = -1
    stepData = []
    dbData = []
    
    try:
        f,b,bt,dbData,stepData = evaluateNPWrapping(np,outFilePath,RUNTIME)
        walk = []
        captured = []
        lost = []
        ps = []
        walkStep = 0
        for s in stepData:
            captured = []
            lost = []
            if walkStep == 0:
                ps = s['walk']
                captured = ps
                walk.append((ps,captured,lost))
                walkStep +=1
                continue
            else:
                sw = s['walk']
                for w in sw:
                    if not w in ps:
                        captured.append(w)
                for w in ps:
                    if not w in sw:
                        lost.append(w)
                walk.append((sw,captured,lost))
                ps = sw
        activeWalk = []
        for w in walk:
            wal = w[0]
            cap = w[1]
            los = w[2]
            activeWal = []
            activeCap = []
            activeLos = []
            for l in wal:
                if np.ligands[l].eps > 1e-4:
                    activeWal.append(l)
            for l in cap:
                if np.ligands[l].eps > 1e-4:
                    activeCap.append(l)
            for l in los:
                if np.ligands[l].eps > 1e-4:
                    activeLos.append(l)
            activeWalk.append((activeWal,activeCap,activeLos))
        wS = 0
        for s in dbData:
            s['walk']=activeWalk[wS]
            wS+=1
    except (OSError, IOError):
        print("Something went wrong...")
        print(outFilePath + ", Wrong file or file path")
        return f,b,bt,dbData
    except:
        print('something unexpected went wrong...')
        return f,b,bt,dbData
    print('{} fitness: {}'.format(simName, f))
    if not KEEPINPUT:
        sim.deleteFiles()
    if KEEPOUTPUT:
        sim.postProcessOutput(outFilePath)
    if os.path.exists(outFilePath):
        try:
            os.remove(outFilePath)
            print "deleted file" + str(outFilePath)
        except (OSError, IOError):
            print "no output file to delete"
    return f,b,bt,dbData

def evaluateParticle(np,simName):

    fitnesses = []
    budding = []
    budTime = []
    stepData = []

    rVecs = []
    rAms = []

    xRA = 1
    yRA = 0

    xR = int(math.ceil(float(REPEATS)/2.0))
    yR = REPEATS - xR

    if xR > 0:
        xRA = 2.0*numpy.pi/float(xR)
    if yR > 0:
        yRA = 2.0*numpy.pi/float(yR)

    for i in range(REPEATS):
        if i%2 ==0:
            rVecs.append([1,0,0])
            rAms.append(float(len(rVecs))*0.5*xRA)
        else:
            rVecs.append([0,1,1])
            rAms.append(float(len(rVecs))*0.5*yRA+numpy.pi*0.5)

    for i in range(REPEATS):
        pf,pb,pbt,psd = evaluateParticleInstance(np,simName+"_"+str(i),rVecs[i],rAms[i])
        fitnesses.append(pf)
        budding.append(pb)
        budTime.append(pbt)
        stepData.append(psd)

    fsum = 0
    for fit in fitnesses:
        fsum+=fit

    budded = True

    for bud in budding:
        if not bud:
            budded = False

    if REPEATS > 0:
        fmem = float(fsum)/float(REPEATS)
    else:
        fmem = float(fsum)

    if budded:
        fmem += BUDDINGREWARD

    #flig = len(np.ligands) 
    #feps = sum([ligand.eps for ligand in np.ligands])

    nActiveLigands = 0
    npTotalEps = 0.0

    for l in np.ligands:
        nActiveLigands += 1
        npTotalEps += l.eps

    if TARGETLIGANDS > 0:
        penalty += TARGETWEIGHT*float(abs(TARGETLIGANDS-nActiveLigands))
        fmem += penalty

    if budded and TARGETLIGANDS < 0 and FIXEDLIGANDS < 0:
        penalty = PENALTYWEIGHT*(1.0-(float(npTotalEps)/(float(EPSMAX)*float(GENES)))) if float(EPSMAX)*float(nActiveLigands) > 0.0 else 0.0
        fmem += penalty
        
    f = fmem

    bTot = 0.0
    bI = 0.0
    bPerc = 0.0

    for b in budding:
        if b:
            bPerc += 1.0

    bPerc = bPerc/len(budding)

    for b in budTime:
        if b != -1:
            bTot += b
            bI += 1

    bAvg = float(bTot)/float(bI) if bI > 0 else -1

    return f,bPerc,bAvg,stepData,fitnesses,budding,budTime



def evaluateBudding(individual):
    phenome = CoveredNanoParticlePhenome(individual,EXPRPLACES,EPSPLACES,EPSMIN,EPSMAX) if not PARTIAL else NanoParticlePhenome(individual,EXPRPLACES,EPSPLACES,POLANGPLACES,AZIANGPLACES,EPSMIN,EPSMAX)
    np = phenome.particle    
    simName = phenome.id + misctools.randomStr(10)
    r = evaluateParticle(np,simName)
    pickleFilePath = os.path.join(OUTDIR,phenome.id+'.pickle')
    if SAVERESULTS:
        if not os.path.exists(pickleFilePath):
            with open(pickleFilePath, 'wb') as handle:
                pickle.dump(r, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return r[0],

def evaluateDiameter(individual):
    minPrune = 0.0
    maxPrune = 1.0
    pruneStep = 0.1
    phenome = CoveredNanoParticlePhenome(individual,EXPRPLACES,EPSPLACES,EPSMIN,EPSMAX) if not PARTIAL else NanoParticlePhenome(individual,EXPRPLACES,EPSPLACES,POLANGPLACES,AZIANGPLACES,EPSMIN,EPSMAX)
    np = phenome.particle
    r = evaluateBudding(individual)
    n = atools.buildLigandNetwork(np.ligands)
    diams = []
    for p in atools.frange(minPrune,maxPrune,pruneStep):
        pN = atools.pruneNetwork(n,p)
        graphs = list(nx.connected_component_subgraphs(pN))
        dS = []
        for g in graphs:
            d = nx.diameter(g)
            dS.append(d)
        diams.append(numpy.max(dS))
    return numpy.mean(diams),

def f2(x,y):
    return numpy.exp(0.1*x) - numpy.exp(0.3*y)

def f3(x,y,z):
    return 25*numpy.exp(-abs(x-7.4)) - numpy.exp(3*abs(y-0.148)) - numpy.exp(0.3*z)

def f4(x,y):
    return 10*numpy.exp(-0.1*abs(x)) + 10*numpy.exp(-0.1*abs(y))

def f5(x,y):
    return 10**numpy.exp(-abs(x)) + numpy.exp(abs(0.1*y))

def evaluateSmallWorld(individual):
    p=0.35
    phenome = CoveredNanoParticlePhenome(individual,EXPRPLACES,EPSPLACES,EPSMIN,EPSMAX) if not PARTIAL else NanoParticlePhenome(individual,EXPRPLACES,EPSPLACES,POLANGPLACES,AZIANGPLACES,EPSMIN,EPSMAX)
    np = phenome.particle
    swf = 100
    f = 0
    n = atools.buildLigandNetwork(np.ligands)
    sws = []
    pN = atools.pruneNetwork(n,p)
    graphs = list(nx.connected_component_subgraphs(pN))
    sw = []
    sigs = []
    maxNodes = 0
    biggestG = None
    spl=1
    totalV=[0,0,0]
    for ligand in np.ligands:
        if ligand.eps > 0.0:
            ligand_v = atools.sphPol2Crt([ligand.rad, ligand.polAng, ligand.aziAng])
            for i,l in enumerate(list(ligand_v)):
                totalV[i]+=l

    for g in graphs:
        if len(g) > maxNodes:
            maxNodes = len(g)
            biggestG = g
        # if biggestG != None and len(biggestG) > 3:
        #     swf = nx.algorithms.smallworld.omega(biggestG)

    if biggestG != None and len(biggestG) > 3:
        swf = nx.diameter(biggestG)
        spl = nx.density(biggestG)

    f = f4(numpy.linalg.norm(totalV),float(len(graphs))) + f5(spl,swf)

    print numpy.linalg.norm(totalV), float(len(graphs)), spl, swf, f

    # try:
    #     for g in graphs:
    #         if len(g) > maxNodes:
    #             maxNodes = len(g)
    #             biggestG = g
    #     # if biggestG != None and len(biggestG) > 3:
    #     #     swf = nx.algorithms.smallworld.omega(biggestG)

    #     if biggestG != None and len(biggestG) > 3:
    #         swf = nx.diameter(biggestG)
    #         spl = nx.density(biggestG)
    #         # swf = nx.density(biggestG)
        
    #     # for g in graphs:
    #     #     if len(g)>8:
    #     #         d = nx.algorithms.smallworld.omega(g)
    #     #         # sig = nx.algorithms.smallworld.sigma(g)
    #     #         if not math.isnan(d):
    #     #             sw.append(d)
    #     #         # sigs.append(sig)
    #     # if len(sw) >0:
    #     #     swf = numpy.mean(sw)
    #         # sigf = numpy.mean(sigs)



    #     # if swf != None and not math.isnan(swf):
    #     #     # f = 100.0/(abs(float(swf))*(float(len(graphs))**4.0))-0.5*float(len(graphs))  if swf != 0.0 else -1E100
    #     #     f = numpy.exp(-(5.0*abs(float(swf))*(float(len(graphs))-0.5) + 0.1*(float(len(graphs))-0.5))) - 0.05*float(len(graphs)) -0.25*abs(float(swf)) + 1.0
    #     f = f3(swf,spl,float(len(graphs)))


    #     # if swf != None and not math.isnan(swf):
    #     #     # f = 100.0/(abs(float(swf))*(float(len(graphs))**4.0))-0.5*float(len(graphs))  if swf != 0.0 else -1E100
    #     #     f = numpy.exp(-(5.0*abs(float(swf+1))*(float(len(graphs))-0.5) + 0.1*(float(len(graphs))-0.5))) - 0.05*float(len(graphs)) -0.25*float(swf) + 1.0
    # except:
    #     print "invalid network, removing individual"
    # if f<0:
    #     f=0
    # if f>15:
    #     f=15
    # print swf,spl,len(graphs), f
    return f,



def sel(pop,k):
    return tools.selTournament(pop,k,TSIZE)

def algorithmEaSimple(pop,toolbox,stats,hof,verbose=VERBOSE):
    return algorithms.eaSimple(pop,toolbox=toolbox,
        cxpb=CXPB, mutpb=MUTPB, ngen=1,verbose=verbose,stats=stats,halloffame=hof)

def operatorDefaultMut(individual):
    return operators.defaultMut(individual, MINPDB)

def operatorFixedActivationMut(individual):
    return operators.fixedActivationMut(individual, MINPDB)

def operatorDefaultGeneWiseTwoPoint(ind1, ind2):
    return operators.defaultGeneWiseTwoPoint(ind1, ind2, GENES, GENESIZE)

def operatorFixedActivationGeneWiseTwoPoint(ind1, ind2):
    return operators.fixedActivationGeneWiseTwoPoint(ind1, ind2, GENES, GENESIZE)

def commitSession(ga):
    novelGenes = []
    dbGen = dao.Generation(ga.gen)
    try:
        iNum = 0
        for isle in ga.islands:
            dbdeme = ga.dbconn.gaSession.demes[iNum]
            for individual in isle:
                np = CoveredNanoParticlePhenome(individual,EXPRPLACES,EPSPLACES,EPSMIN,EPSMAX) if not PARTIAL else NanoParticlePhenome(individual,EXPRPLACES,EPSPLACES,POLANGPLACES,AZIANGPLACES,EPSMIN,EPSMAX)
                budData = None
                pickleFile = os.path.join(OUTDIR,np.id+'.pickle')
                try:
                    with open(pickleFile, 'rb') as handle:
                        budData = pickle.load(handle)
                except:
                    print "no bud data"

                i = dao.Individual(individual, np)
                i.deme = dbdeme
                if budData != None:
                    i.budPerc = budData[1]
                    i.budTime = budData[2]
                for simData in budData[3]:
                    if not ga.dbconn.ifSimExists(simData):
                        s = dao.Simulation()
                        s.data = simData
                        i.sims.append(s)
                    else:
                        s = ga.dbconn.getSimByData(simData)
                        i.sims.append(s)
                dbGen.individuals.append(i)
                for g in np.genelist:
                    gene = dao.Gene(g)
                    novelGene = True
                    for n in dbGen.novelGenes:
                        if n.rawGene == gene.rawGene:
                            novelGene = False
                            i.addGene(n)
                    if novelGene:
                        for gen in ga.dbconn.gaSession.generations:
                                for nGene in range(len(gen.novelGenes)):
                                    if gen.novelGenes[nGene].rawGene == gene.rawGene:
                                        novelGene = False
                                        i.addGene(gen.novelGenes[nGene])
                                        break
                    if novelGene:
                        dbGene = ga.dbconn.getGeneByRawGene(gene.rawGene)
                        if dbGene != None:
                            novelGene = False
                            i.addGene(dbGene)
                    if novelGene:
                        dbGen.novelGenes.append(gene)
                        i.addGene(gene)
            iNum += 1

        ga.dbconn.saveGeneration(dbGen)

        ga.dbconn.gaSession.metrics.metricsPickle = ga.metrics
        ga.dbconn.gaSession.genealogy.treePickle = ga.history.genealogy_tree
        ga.dbconn.gaSession.genealogy.historyPickle = ga.history.genealogy_history

        ga.dbconn.commit()
    except IOError as e:
        print "I/O error({0}): {1}".format(e.errno, e.strerror)
    except Exception as e:
        print "Unexpected error {0}".format(str(e))

def beforeMigration(ga):
    misctools.removeByPattern(WDIR,"subhedra")

    return

def afterMigration(ga):
    if SAVERESULTS:
        commitSession(ga)

    if KEEPBEST:
        saveBest(ga.islands,ga.gen)

    if KEEPWORST:
        saveWorst(ga.islands,ga.gen)

    return

def saveBest(pop,gen):
    if len(pop) < 1:
        return
    if len(pop[0]) < 1:
        return
    ind = pop[0][0]
    maxFit = ind.fitness.values[-1]
    for isle in pop:
        for p in isle:
            if p.fitness.values[-1] > maxFit:
                ind = p
                maxFit = ind.fitness.values[-1]
    print "saving best:" + str(ind)+ ", fitness:" + str(ind.fitness.values[-1])
    phenome = CoveredNanoParticlePhenome(ind,EXPRPLACES,EPSPLACES,EPSMIN,EPSMAX) if not PARTIAL else NanoParticlePhenome(ind,EXPRPLACES,EPSPLACES,POLANGPLACES,AZIANGPLACES,EPSMIN,EPSMAX)
    np = phenome.particle
    sim = MembraneSimulation(
        'gen_'+str(gen)+"_best",
        np,
        25000,
        TIMESTEP,            
        OUTDIR,
        HOFDIR,            
        TEMPLATEDATAPATH,
        TEMPLATEINPUTPATH,
        rAxis = vectools.randomUnitVector(),
        rAmount = random.uniform(0.3141,3.141)
        )
    hofScriptPath = os.path.join(sim.filedir,sim.scriptName)
    sim.saveFiles()
    runSim(hofScriptPath)
    outFilePath = os.path.join(sim.outdir,sim.outName)
    sim.postProcessOutput(outFilePath)

def saveWorst(pop,gen):
    if len(pop) < 1:
        return
    if len(pop[0]) < 1:
        return
    minFit = 1e8
    ind = pop[0][0]
    for isle in pop:
        for p in isle:
            if p.fitness.values[-1] < minFit:
                ind = p
                minFit = ind.fitness.values[-1]
    print "saving worst:" + str(ind)+ ", fitness:" + str(ind.fitness.values[-1])
    phenome = CoveredNanoParticlePhenome(ind,EXPRPLACES,EPSPLACES,EPSMIN,EPSMAX) if not PARTIAL else NanoParticlePhenome(ind,EXPRPLACES,EPSPLACES,POLANGPLACES,AZIANGPLACES,EPSMIN,EPSMAX)
    np = phenome.particle
    sim = MembraneSimulation(
        'gen_'+str(gen)+"_worst",
        np,
        25000,
        TIMESTEP,            
        OUTDIR,
        HOFDIR,            
        TEMPLATEDATAPATH,
        TEMPLATEINPUTPATH,
        rAxis = vectools.randomUnitVector(),
        rAmount = random.uniform(0.3141,3.141)
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
            25000,
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
        i+=1

def mkdirs(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def mEval(tu):
    return (tu[1],tu[0],evaluateParticle(tu[0],tu[1]))

def main():

    if ZOO != None:
        if os.path.isfile(ZOO) and os.access(ZOO, os.R_OK):
            print "Loading zoo pickle file..."
        else:
            print "Either the file is missing or not readable"
            return
        zooList = {}
        try:
            with open(ZOO, 'rb') as handle:
                zooList = pickle.load(handle)
        except:
            print "Something went wrong reading the zoo file"
            return
        pmap = pools.ProcessPool(WORKERS).map
        results = pmap(mEval,[(v,k) for k,v in zooList.iteritems()])

        #serialised
        # results = []
        # for k,v in zooList.iteritems():
        #     results.append(mEval((v,k)))
        with open(os.path.join(DBDIR,ZOO.split('.')[0]+'.csv'), "a") as csv:
            for r in results:
                csv.write(str(r[0]))
                csv.write(','+str(len(r[1].ligands)))
                csv.write(','+str(numpy.mean([i.eps for i in r[1].ligands])))
                for i in range(3):
                    csv.write(','+str(r[2][i]))
                csv.write('\n')
        return




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
        
    if runArgs.graph == 'singlet':
       network = networks.createSinglets(NISLES)
    elif runArgs.graph == 'islands':
       network = networks.createIslands(NISLES)
    elif runArgs.graph == 'star':
       network = networks.createStar(NISLES-1)
    elif runArgs.graph == 'megastar':
       network = networks.createMegaStar(NSPOKES,int(math.ceil((NISLES-3)*0.25)),int(math.floor((NISLES-3)*0.25)))
    else:
       raw_input('malformed network option, continue with islands? (Enter)')

    if runArgs.algorithm == 'eaSimple':
        algorithm = algorithmEaSimple
    else:
        raw_input('malformed algorithm option, continuing with eaSimple.. (Enter)')
        algorithm = algorithmEaSimple

    if runArgs.mutation == 'defaultMut' and runArgs.fixedligands == -1:
        mut = operatorDefaultMut
    elif runArgs.mutation == 'fixedActivationMut' or runArgs.fixedligands > -1:
        mut = operatorFixedActivationMut
    else:
        raw_input('malformed mutation option, continuing with defaultMut.. (Enter)')
        mut = operatorDefaultMut

    if runArgs.mate == 'defaultGeneWiseTwoPoint' and runArgs.fixedligands == -1:
        mate = operatorDefaultGeneWiseTwoPoint
    elif runArgs.mate == 'fixedActivationGeneWiseTwoPoint' or runArgs.fixedligands > -1:
        mate = operatorFixedActivationGeneWiseTwoPoint
    else:
        raw_input('malformed mate option, continuing with defaultGeneWiseTwoPoint.. (Enter)')
        mate = operatorDefaultGeneWiseTwoPoint

    random.seed(runArgs.seed)

    if SAVERESULTS:
       dbconn = dao.DatabaseConnection(DBPATH)
       dbconn.saveSession(args)
       dbconn.commit()
    else:
       dbconn = None

    initPopFile = "init.json" if FILE == None else FILE

    evaluate = None

    if FITNESSFUNCTION == 'budding':
        evaluate = evaluateBudding
    elif FITNESSFUNCTION == 'diameter':
        evaluate = evaluateDiameter
    elif FITNESSFUNCTION == 'smallworld':
        evaluate = evaluateSmallWorld


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
       mate = mate,
       dbconn = dbconn,
       jsonFile = initPopFile,
       loadFromFile = LOADFROMFILE,
       fixedLigands = runArgs.fixedligands
       )

    

    if SAVERESULTS:
        dbconn.gaSession.metrics = dao.Metrics(ga.metrics)
        ga.dbconn.gaSession.genealogy = dao.Genealogy(ga.history.genealogy_tree,ga.history.genealogy_history)
        for isle in ga.islands:
            ga.dbconn.gaSession.demes.append(dao.Deme())
        ga.dbconn.commit()

    ga.firstGeneration()

    if SAVERESULTS:
        commitSession(ga)

    results = ga.run(NGEN,FREQ,MIGR,STARTINGGEN)

    saveMetrics(results[-2])
    saveHOF(results[1])

    if SAVERESULTS:
        try:
           ga.dbconn.commit()
           ga.dbconn.close()
           misctools.removeByPattern(OUTDIR,"pickle")
        except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)
        except:
            print "Unexpected error:", sys.exc_info()[0]                    


if __name__ == "__main__":
    main()