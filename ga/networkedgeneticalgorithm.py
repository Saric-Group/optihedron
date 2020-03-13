import array
import random
import math

import itertools
import operator

import numpy

from . import networks

from deap import base
from deap import creator
from deap import tools

from pprint import pprint

import json

import operators


from ga import algorithms

from pathos import pools

import time


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", numpy.ndarray, fitness=creator.FitnessMax) # @UndefinedVariable (for PyDev)

def defaultAlgorithmEaSimple(pop,toolbox,stats,hof):
		return algorithms.eaSimple(pop,toolbox=toolbox,
		cxpb=0.5, mutpb=0.2, ngen=1,verbose=False,stats=stats,halloffame=hof)

def defaultTwoPoint(ind1, ind2):
    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else: # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()
        
    return ind1, ind2

def defaultEvalMax(individual):
    return sum(individual),

def defaultSelTournament(pop,k):
    return tools.selTournament(pop,k,3)

def defaultMutFlipBit(individual):
    return tools.mutFlipBit(individual,0.05)

def accumulateStats(l):
    acc = []
    for key, group in itertools.groupby(l, lambda item: item["gen"]):
        glist = list(group)
        acc.append({'gen': key,
            'max':  numpy.max([item["max"] for item in glist])
            ,'min':  numpy.min([item["min"] for item in glist])
            ,'avg':    numpy.mean([item["avg"] for item in glist])
            ,'std':    math.sqrt(numpy.sum(map(lambda x: x*x,[item["std"] for item in glist])))
            })
    return acc


class NetworkedGeneticAlgorithm:

    def __init__(
    	self,
    	genomeSize,
    	islePop,
        hofSize,
    	evaluate = defaultEvalMax,
    	sel = defaultSelTournament,
    	net = networks.createIslands(10),
    	subroutine = defaultAlgorithmEaSimple,
    	mate = defaultTwoPoint,
    	mut = defaultMutFlipBit,
        mapping = map,
    	beforeMigration=lambda x: None,
    	afterMigration=lambda x: None,
        verbose = False,
        dbconn = None,
        jsonFile = "init.json",
        loadFromFile = False,
        fixedLigands = -1
        ):

        
        self.toolbox = base.Toolbox()
        self.history = tools.History()

        self.jsonFile = jsonFile
        self.loadFromFile = loadFromFile

        # Attribute generator
        self.toolbox.register("attr_bool", random.randint, 0, 1)

        # Structure initializers
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_bool, genomeSize)  # @UndefinedVariable (for PyDev)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("individual_guess", self.initIndividual, creator.Individual)
        self.toolbox.register("population_guess", self.initPopulation, list, self.toolbox.individual_guess, self.jsonFile)

        self.toolbox.register("evaluate", evaluate)
        self.toolbox.register("mate", mate)
        self.toolbox.register("mutate", mut)
        self.toolbox.register("select", sel)
        self.toolbox.register("map", mapping)

        self.toolbox.decorate("mate", self.history.decorator)
        self.toolbox.decorate("mutate", self.history.decorator)

        self.net = net
        self.subroutine = subroutine
        self.islePop = islePop
        self.beforeMigration = beforeMigration
        self.afterMigration = afterMigration
        self.hof = self.buildHOF(hofSize)
        self.verbose = verbose
        self.dbconn = dbconn
        self.gen = 0
        self.novelty=[]
        self.metrics = []
        self.islands = self.toolbox.population_guess() if self.loadFromFile else [self.toolbox.population(n=self.islePop) for i in range(len(self.net))]

        if fixedLigands > -1:
            print "overriding population based on fixed ligand count of " + str(fixedLigands)
            for isle in self.islands:
                for ind in range(len(isle)):
                    isle[ind] = operators.fixActivation(isle[ind],fixedLigands)


    def initIndividual(self,icls,content):
        return icls(content)

    def initPopulation(self,plcs,ind_init,filename):
        try:
            with open(filename, "r") as pop_file:
                contents = json.load(pop_file)
            demes = contents['init_pop']
            iPop = []
            for d in demes:
                iPop.append([ind_init(i) for i in d])
            return iPop
        except:
            print("error loading " + filename)
            return []

    def genMetrics(self,gen,island,logbook):
        log = []
        for d in logbook:
            d['gen']+=gen
            d['island']=island
            log.append(d)
        return log

    def getMinFitness(self,pop):
        return numpy.min([p.fitness.values[-1] for p in pop])

    def getTotalFitness(self,pop):
        return numpy.sum([p.fitness.values[-1] for p in pop])


    def migration(self,islands):
        if len(islands)<2:
            return islands
        network = self.net
        minF = numpy.min(list(map(self.getMinFitness,islands)))
        rFitness = 1.0/(numpy.sum(list(map(self.getTotalFitness,islands))))
        s = random.uniform(0,1)
        t = 0.0
        for i in range(len(islands)):
            isle = islands[i]
            for ind in isle:
                f = ind.fitness.values[-1]*rFitness
                if f + t >= s:
                    edges = network.edges(list(network.nodes())[i])
                    if len(edges)>0:
                        edge = random.choice(list(edges))[-1]
                        dest = list(network.nodes()).index(edge)
                        #print("migrating " + str(ind) + " from " + str(i)  + " to " + str(dest))
                        islands[dest][random.randint(0,len(islands[dest])-1)] = ind
                    return islands
                else:
                    t+=f
        return islands

    def algorithm(self,pop,verbose=False):
        return self.subroutine(pop,self.toolbox,self.buildStats(),self.hof,verbose=verbose)

    def buildStats(self):
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean)
        stats.register("std", numpy.std)
        stats.register("min", numpy.min)
        stats.register("max", numpy.max)
        return stats

    def buildHOF(self,size=1):
    	return tools.HallOfFame(size, similar=numpy.array_equal)

    def evaluateIndividuals(self,population):

        stats = self.buildStats()
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate,population)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if self.hof is not None:
            self.hof.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)

        return population, logbook

    def firstGeneration(self):
        self.results = map(self.evaluateIndividuals,self.islands)
        self.metrics += map(self.genMetrics,[0]*len(list(self.results)),[n for n in range(len(list(self.results)))], [logbook for pop, logbook in self.results])
        for isle in self.islands:
            self.history.update(isle)
        
    def run(self,ngen,freq,migr,startingGen=0):
        self.metrics = []
        pool = pools.ProcessPool(10)
        for i in range(startingGen, ngen):
            self.gen = i+1
            if self.verbose:
                print("GEN: " + str(i+1) + "/" + str(ngen))
            self.results = map(self.algorithm, self.islands)
            self.islands = [pop for pop, logbook in self.results]
            self.metrics += map(self.genMetrics,[i]*len(list(self.results)),[n for n in range(len(list(self.results)))], [logbook for pop, logbook in self.results])
            for isle in self.islands:
                self.history.update(isle)
            self.beforeMigration(self)
            if i%freq == 0 and self.gen < ngen - 1:
                for i in range(0,migr):
                	self.islands = self.migration(self.islands)
            self.afterMigration(self)
        self.metrics = [val for sublist in self.metrics for val in sublist]
        self.metrics = sorted(sorted(self.metrics, key=lambda k: k['island']), key=lambda k: k['gen']) 
        self.accMetrics = (accumulateStats(self.metrics))
        #self.metrics = list(self.accumulate(self.metrics))
        return self.islands, self.hof, self.metrics, self.accMetrics, self.history





