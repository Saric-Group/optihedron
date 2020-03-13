import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import os
import copy
import time
import pickle
import networkx
import holoviews as hv
import csv, codecs, cStringIO
import sys
import math
import random
from itertools import cycle
from collections import namedtuple
import pandas as pd
from scipy.spatial import distance_matrix


def buildNanoParticleFromNetwork(G,weight,radius=4,sig=1):
    particle = nanoparticle.NanoParticle()
    for n,w in G.nodes(data=True):
        particle.ligands.append(nanoparticle.Ligand(weight,sig,radius,w['pol'],w['azi']))
    return particle


def buildLigandNetwork(ligands, silent=True, ignoreZeros=True):
    if not silent:
        startTime = time.time()
    G=networkx.Graph()
    
    nIndex = 1
    for i in ligands:
        if i.eps > 0.0:
            G.add_node(nIndex,weight=i.eps,polAng=i.polAng,aziAng=i.aziAng)
        elif not ignoreZeros:
            G.add_node(nIndex,weight=i.eps,polAng=i.polAng,aziAng=i.aziAng)
        nIndex += 1
    
    iIndex = 1
    for i in ligands:
        jIndex = 1
        for j in ligands:
            if i < j:
                #affDist = abs(i.eps - j.eps)
                if i.eps > 0.0 and j.eps > 0.0:
                    cartDist = 1.0/greatArcDist((i.polAng,i.aziAng),(j.polAng,j.aziAng))
                    G.add_edge(iIndex, jIndex, weight=cartDist)
            jIndex += 1
        iIndex += 1
    return G

def pruneNetwork(G,pruning,pruneNodes=True,pruneEdges=True):
    prunes = []
    pruneNodes = []
    GP = copy.deepcopy(G)
    maxW = 0
    if pruneNodes and len(G.nodes()) > 0:
        for n,w in G.nodes(data=True):
            if 'weight' not in w or w['weight'] <= 0.0:
                pruneNodes.append(n)

    if pruneEdges and len(G.edges()) > 0:
        for e in GP.edges:
            w = GP.get_edge_data(*e)['weight']
            if(w<=pruning):
                prunes.append(e)

    GP.remove_edges_from(prunes)
    GP.remove_nodes_from(pruneNodes)
    return GP


def buildPrunedLigandNetwork(ind,pruning):
    G = buildLigandNetwork(ind.phenomePickle.particle.ligands)
    return pruneNetwork(G,pruning)

def buildPrunedNetworkView(ind,pruning):
    G = buildPrunedLigandNetwork(ind,pruning)
    padding = dict(x=(-1.2, 1.2), y=(-1.2, 1.2))
    return hv.Graph.from_networkx(G, networkx.layout.spring_layout).opts(plot=dict(color_index='weight')).redim.range(**padding)

def buildNetworkView(G):
    padding = dict(x=(-1.2, 1.2), y=(-1.2, 1.2))
    return hv.Graph.from_networkx(G, networkx.layout.spring_layout).opts(plot=dict(color_index='weight')).redim.range(**padding)

def hammingDistance(s1,s2):
    assert len(s1) == len(s2)
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def formFactor(ligands):
    
    spottyLigands = 0    
    lineyLigands = 0    
    patchyLigands = 0
    
    minDist = 2.0
    smallestDist = 100

    #counting liney ligands
    nZero = 0
    for j in ligands:
        if j.eps == 0.0:
            nZero+=1
            continue
        NNlist = []
        for k in ligands:        
            if j != k and k.eps > 0.0:
                dist = greatArcDist((j.polAng,j.aziAng),(k.polAng,k.aziAng))
                if dist <= minDist:
                    NNlist.append(k)
                if dist < smallestDist:
                    smallestDist = dist
        NNcount = len(NNlist)
        if NNcount == 2:#if it has 2 NNs 
            dist = greatArcDist((NNlist[0].polAng,NNlist[0].aziAng),(NNlist[1].polAng,NNlist[1].aziAng))
            if not dist <= minDist:#and the NNs are not each other's NN
                lineyLigands += 1                              
        elif NNcount == 1:
            lineyLigands += 1 
        elif NNcount == 0:
            spottyLigands += 1
    
    #counting patchy ligands
    totalLigands = float(len(ligands)) - nZero
    patchyLigands = totalLigands - spottyLigands - lineyLigands
    
    return [float(patchyLigands)/totalLigands, float(lineyLigands)/totalLigands, float(spottyLigands)/totalLigands]

def buildNetworkList(individuals):
    networks = []
    for i in individuals:
        n = buildLigandNetwork(i.phenomePickle.particle.ligands)
        networks.append((n,i))
    return networks

def pruneNetworkList(networks,pruning):
    prunedNetworks = []
    for n in networks:
        prunedNetworks.append((pruneNetwork(n[0],pruning),n[1]))
    return prunedNetworks

def buildPrunedNetworkList(individuals,pruning):
    return pruneNetworkList(buildNetworkList(individuals),pruning)

def gnmw(G):
    return(len(G.nodes()),len(G.edges()),np.sum([w['weight'] for (u,v,w) in G.edges(data=True)]))

def gnmRandomWeightedGraph(nodes,edges,weight):
    G = networkx.gnm_random_graph(nodes,edges)
    totalWeight = 0
    for (u,v,w) in G.edges(data=True):
        w['weight'] = random.uniform(0,10)
        totalWeight += w['weight']
    
    normFactor = weight/totalWeight
    
    for (u,v,w) in G.edges(data=True):
        w['weight'] = w['weight']*normFactor
        
    return G

def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"

    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0, 0.8, N+4)
    return mycmap

def smallWorldNess(G):
    gnmwG = gnmw(G)
    attempts = 1
    RG = gnmRandomWeightedGraph(gnmwG[0],gnmwG[1],gnmwG[2])
    while len(list(networkx.connected_component_subgraphs(RG))) > 1 and attempts < 10:
        RG = gnmRandomWeightedGraph(gnmwG[0],gnmwG[1],gnmwG[2])
        attempts += 1
    if len(list(networkx.connected_component_subgraphs(RG))) > 1:
        return 0
    pLengthG = networkx.average_shortest_path_length(G,weight='weight')
    pLengthRG = networkx.average_shortest_path_length(RG,weight='weight')
    if pLengthRG <= 0 or pLengthG <= 0:
        return 0
    clusteringG = networkx.average_clustering(G,weight='weight')
    clusteringRG = networkx.average_clustering(RG,weight='weight')
    if clusteringRG <= 0:
        return 0
    SW = (clusteringG/clusteringRG)/(pLengthG/pLengthRG)
    return SW

def smallWorld2(G):
    return networkx.algorithms.smallworld.omega(G)

def dropChildren(data, parentKey, childKeys, silent=True):
	if not silent:
		print('removing {} from {}!'.format(childKeys, parentKey))
	for parentItem in data[parentKey]:
		for childKey in childKeys:
			del parentItem[childKey]
	return data

def dropParentsConditional(data, parentKey, condition):
	data[parentKey] = [i for i in data[parentKey] if condition(i)]
	return data

def dropParents(data, parentKey):
	del data[parentKey]
	return data

def load(DBs, dbConnect, clean = None, drop = None, sort = None):
	data = []

	for DB in DBs:
		dbConn = dbConnect.DatabaseConnection(DB)
		datum = dbConn.loadSession('1')
		
		if clean:
			for task in clean:
				datum = task(datum)
		if drop:
			for task in drop:
				if isinstance(task, basestring):
					del datum[task]
				else:
					datum = task(datum)
		if sort:
			for task in sort:
				datum = task(datum)

		data.append(datum)
		dbConn.close()
		
	return data


def cleanLigands(data):
    for ind in data['individuals']:
        ind['phenome'].particle.ligands = [lig for lig in ind['phenome'].particle.ligands if lig.eps != 0]
    return data

def cleanFits(data):
    for ind in data['individuals']:
        if ind['fitness'] > 400.0:            
            ind['fitness'] = 400.0 + 600.0*(1.0-((np.sum([lig.eps for lig in ind['phenome'].particle.ligands]))/(15.0*72.0)))            
    return data

def sortIndLigandCount(data):
    data['individuals'].sort(key = lambda ind: len(ind['phenome'].particle.ligands))
    return data

def sortIndGens(data):
    data['individuals'].sort(key = lambda ind: ind['gen'])
    return data

def sortIndFits(data):
    data['individuals'].sort(key = lambda ind: ind['fitness'])
    return data

def sameLigands(ligA, ligB):
	if ligA.__dict__ == ligB.__dict__:
		return True
	else:
		return False



def greatArcDist(Ang1, Ang2, rad=4):
    #Ang = (PolarAng,AziAng)
    #https://math.stackexchange.com/questions/231221/great-arc-distance-between-two-points-on-a-unit-sphere
    arcDist=rad*(np.arccos((np.cos(Ang1[0])*np.cos(Ang2[0]))+((np.sin(Ang1[0])*np.sin(Ang2[0]))*(np.cos(Ang1[1]-Ang2[1])))))
    return arcDist

def tagLigands(data):
	data['individuals'] = [classifyLigands(ind) for ind in data['individuals']]
	return data		

def classifyLigands(ind):
    cutoff = 2.0

    if 'ligtags' in ind:
		return ind['ligtags']
    else:
        ligands = ind['phenome'].particle.ligands
        spotty = 0
        liney = 0
        patchy = 0
        identity = {}

        for ligCursor in ligands:	
            identity[ligCursor] = {}
            NNcount = 0
            NNlist = []

            for ligOther in ligands:
                if not sameLigands(ligCursor, ligOther):
                    arcDist = greatArcDist((ligCursor.polAng,ligCursor.aziAng),(ligOther.polAng,ligOther.aziAng)) 
                    if arcDist <= cutoff:
                        NNcount += 1
                        NNlist.append(ligOther)

            identity[ligCursor]['NNcount'] = NNcount
            identity[ligCursor]['NNlist'] = NNlist

            if NNcount == 3: #if 3 NN
                arcDist12 = greatArcDist((NNlist[0].polAng,NNlist[0].aziAng),(NNlist[1].polAng,NNlist[1].aziAng))
                arcDist13 = greatArcDist((NNlist[0].polAng,NNlist[0].aziAng),(NNlist[2].polAng,NNlist[2].aziAng))
                arcDist23 = greatArcDist((NNlist[1].polAng,NNlist[1].aziAng),(NNlist[2].polAng,NNlist[2].aziAng))

                if not arcDist12 <= 2.0 and not arcDist13 <= 2.0 and not arcDist23 <= 2.0: #and noone of them are ea other's NN                    
                    liney += 1
                    identity[ligCursor]['character'] = 'liney'

            if NNcount == 2:#if 2 NN                
                arcDist = greatArcDist((NNlist[0].polAng,NNlist[0].aziAng),(NNlist[1].polAng,NNlist[1].aziAng))
                if not arcDist <= 2.0:#and the NNs are not each other's NN
                    liney += 1
                    identity[ligCursor]['character'] = 'liney'
          
            if NNcount == 1:#if 1 NN
                liney += 1
                identity[ligCursor]['character'] = 'liney'

            if NNcount == 0: #if 0 NN
                spotty += 1
                identity[ligCursor]['character'] = 'spotty'
            
            #all other ligands contribute to overall patchy-ness            
            if 'character' not in identity[ligCursor]:
                identity[ligCursor]['character'] = 'patchy'

        patchy = len(ligands) - spotty - liney
        ind['ligtags'] = patchy, spotty, liney, identity

        return patchy, spotty, liney, identity

def clusterLineyLigands(ligands, silent=True):
    #print(greatArcDist((0,0), ((3.14/15.0),0))) #bad at small distances, correct should be 0.8377 but returns 0.8373, willing to accept +- 0.001
    #ligands = [(ligand,ligandTags)]
    if not silent:
        startTime = time.time()
    
    ligandsTmp = copy.deepcopy(ligands)

    clusters = []
    while ligandsTmp:            
        nextSeedQueue = [ligandsTmp[0]]    

        clusterTmp = []
        clusterTmp.append(ligandsTmp[0])
        del ligandsTmp[0]
        while nextSeedQueue:
            seed = nextSeedQueue.pop()
            rLigands = []
            for ligand in ligandsTmp:
                NNlist = ligand[1]['NNlist']
                existingCopies = 0
                for NN in NNlist:
                    if sameLigands(seed[0], NN):
                        existingCopies += 1
                if existingCopies == 1:                                    
                    nextSeedQueue.append(ligand)
                    clusterTmp.append(ligand)
                    rLigands.append(ligand)
                elif existingCopies > 1:
                    raise ValueError
            for rLig in rLigands:
                ligandsTmp.remove(rLig)
        clusters.append(clusterTmp)        

    if not silent:
        print('ligands: {}'.format(len(ligands)))
        print('clusters: {}'.format(len(clusters)))
        print('clustered in {}s'.format(time.time()-startTime))

    return clusters

def groupLineyLigands(ind):
    if 'lineyclustertags' in ind:
        return ind['lineyclustertags']
    else:
        _, _, _, identity = classifyLigands(ind)
        lineyLigands = []

        for ligand, ligandTags in identity.items():
            if ligandTags['character'] == 'liney':
                lineyLigands.append((ligand,ligandTags)) 
        lineyClusters = clusterLineyLigands(lineyLigands)
        ind['lineyclustertags'] = lineyClusters
        return lineyClusters

def scanGen(scanData, interest, indexOffset, aggregateMode, silent=True):
    scanPlotIndex = []
    scanPlotData = []

    aggregateMethod = None
    if aggregateMode != 'MIN' and aggregateMode != 'MAX' and aggregateMode != 'AVG' and aggregateMode != 'POP':
        raise ValueError    

    if not silent:
        startTime = time.time()
    for scanDatum in scanData:

        aggregateInterest = []
        aggregateBucket = []
        aggregateInds = 0
        cursorGen = 0
        genAveragedInterest = []        

        expectedTicks = len(scanDatum['individuals'])
        actualTicks = 0

        for ind in scanDatum['individuals']:            
            if ind['gen'] != cursorGen:    
                if not silent:
                    print('analyzed gen{}'.format(cursorGen))

                if aggregateMode == 'MIN':
                    genAveragedInterest.append(((cursorGen),np.min(aggregateInterest)))
                elif aggregateMode == 'MAX':
                    genAveragedInterest.append(((cursorGen),np.max(aggregateInterest)))
                elif aggregateMode == 'AVG':
                    genAveragedInterest.append(((cursorGen),(np.sum(aggregateInterest)/float(aggregateInds))))
                elif aggregateMode == 'POP':
                    genAveragedInterest.append(((cursorGen),interest(aggregateBucket)))
                
                actualTicks += aggregateInds
                cursorGen = ind['gen']                    
                if aggregateMode == 'MIN' or aggregateMode == 'MAX' or aggregateMode == 'AVG':
                    aggregateInterest = [interest(ind)]
                elif aggregateMode == 'POP':
                    aggregateBucket = [ind]
                aggregateInds = 1
            else:                
                if aggregateMode == 'MIN' or aggregateMode == 'MAX' or aggregateMode == 'AVG':
                    aggregateInterest.append(interest(ind)) 
                elif aggregateMode == 'POP':
                    aggregateBucket.append(ind)
                aggregateInds += 1

        if aggregateMode == 'MIN':
            genAveragedInterest.append(((cursorGen),np.min(aggregateInterest)))
        elif aggregateMode == 'MAX':
            genAveragedInterest.append(((cursorGen),np.max(aggregateInterest)))
        elif aggregateMode == 'AVG':
            genAveragedInterest.append(((cursorGen),(np.sum(aggregateInterest)/float(aggregateInds))))
        elif aggregateMode == 'POP':
            genAveragedInterest.append(((cursorGen),interest(aggregateBucket)))
        actualTicks += aggregateInds

        if expectedTicks != actualTicks:
            raise ValueError

        scanPlotIndex.append(indexOffset)
        scanPlotData.append(genAveragedInterest)            
        if not silent:
            print('analyzed gen{}'.format(cursorGen))
            print('analyzed index {} of scanData'.format(indexOffset))
            print('analysis took {}s'.format(time.time() - startTime))            
        indexOffset += 1
    return scanPlotIndex, scanPlotData

def scanCustom(scanData, interestKey, interestKeyLabel, tickerRange, tickerBlockSize, tickerBlockOffset, interest, indexOffset, aggregateMode, tickerInterval = 0.0, silent=True):
    scanPlotIndex = []
    scanPlotData = []    

    aggregateMethod = None
    if aggregateMode != 'MIN' and aggregateMode != 'MAX' and aggregateMode != 'AVG' and aggregateMode != 'POP':
        raise ValueError    

    if not silent:
        startTime = time.time()
    for scanDatum in scanData:

        ticks = []
        for tickerBlock in range(tickerRange):
            ticks.append(tickerBlock*tickerBlockSize+tickerBlockOffset)        

        aggregateInterest = []
        aggregateBucket = []
        aggregateInds = 0
        cursorTick = ticks.pop(0)
        tickAveragedInterest = []        

        expectedTicks = len(scanDatum['individuals'])
        actualTicks = 0        

        for ind in scanDatum['individuals']:                                   
            if interestKey(ind) - cursorTick > tickerInterval:                    
                if not silent:
                    print('analyzed tick {} of {}'.format(cursorTick, interestKeyLabel))

                if aggregateInds != 0:
                    if aggregateMode == 'MIN':
                        tickAveragedInterest.append(((cursorTick),np.min(aggregateInterest)))
                    elif aggregateMode == 'MAX':
                        tickAveragedInterest.append(((cursorTick),np.max(aggregateInterest)))
                    elif aggregateMode == 'AVG':
                        tickAveragedInterest.append(((cursorTick),(np.sum(aggregateInterest)/float(aggregateInds))))
                    elif aggregateMode == 'POP':
                        tickAveragedInterest.append(((cursorTick),interest(aggregateBucket)))
                    
                    actualTicks += aggregateInds
                    cursorTick = ticks.pop(0)

                while interestKey(ind) - cursorTick > tickerInterval:                        
                    tickAveragedInterest.append(((cursorTick),np.nan))
                    cursorTick = ticks.pop(0)
                
                if aggregateMode == 'MIN' or aggregateMode == 'MAX' or aggregateMode == 'AVG':
                    aggregateInterest = [interest(ind)]
                elif aggregateMode == 'POP':
                    aggregateBucket = [ind]
                aggregateInds = 1
            else:
                if aggregateMode == 'MIN' or aggregateMode == 'MAX' or aggregateMode == 'AVG':
                    aggregateInterest.append(interest(ind)) 
                elif aggregateMode == 'POP':
                    aggregateBucket.append(ind)                                
                aggregateInds += 1

        if aggregateMode == 'MIN':
            tickAveragedInterest.append(((cursorTick),np.min(aggregateInterest)))
        elif aggregateMode == 'MAX':
            tickAveragedInterest.append(((cursorTick),np.max(aggregateInterest)))
        elif aggregateMode == 'AVG':
            tickAveragedInterest.append(((cursorTick),(np.sum(aggregateInterest)/float(aggregateInds))))
        elif aggregateMode == 'POP':
            tickAveragedInterest.append(((cursorTick),interest(aggregateBucket)))
        actualTicks += aggregateInds

        if expectedTicks != actualTicks:            
            raise ValueError

        while ticks:
            tickAveragedInterest.append(((ticks.pop(0)),np.nan))

        scanPlotIndex.append(indexOffset)
        scanPlotData.append(tickAveragedInterest)            
        if not silent:
            print('analyzed tick {} of {}'.format(cursorTick, interestKeyLabel))
            print('analyzed index {} of scanData'.format(indexOffset))
            print('analysis took {}s'.format(time.time() - startTime))            
        indexOffset += 1
    return scanPlotIndex, scanPlotData

def characterScore(ind, patchyScore = -1, spottyScore = -1, lineyScore = 1):
    patchy, spotty, liney, _ = classifyLigands(ind)
    charScore = float((patchy*patchyScore) + (spotty*spottyScore) + (liney*lineyScore))/float(patchy+spotty+liney)

    return charScore

def lineyABS(ind):
    _, _, liney, _ = classifyLigands(ind)

    return float(liney)

def spottyABS(ind):
    _, spotty, _, _ = classifyLigands(ind)

    return float(spotty)

def patchyABS(ind):
    patchy, _, _, _ = classifyLigands(ind)

    return float(patchy)

def lineyREL(ind):
    patchy, spotty, liney, _ = classifyLigands(ind)

    return float(liney)/float(patchy+spotty+liney)

def spottyREL(ind):
    patchy, spotty, liney, _ = classifyLigands(ind)

    return float(spotty)/float(patchy+spotty+liney)

def patchyREL(ind):
    patchy, spotty, liney, _ = classifyLigands(ind)

    return float(patchy)/float(patchy+spotty+liney)

def NNCountAVG(ind):
    _, _, _, identity = classifyLigands(ind)
    totalNNs = 0
    totalLigands = 0
    for ligand, ligandTags in identity.items():
        totalNNs += ligandTags['NNcount']
        totalLigands += 1
    return float(totalNNs)/float(totalLigands)

def lineyLineTags(ind, minLineLength):      
    if 'lineylinetags' in ind:
        return ind['lineylinetags']
    else:        
        lineyClusters = groupLineyLigands(ind)
        lineyLigands = []
        for cluster in lineyClusters:
            for ligand, ligandTags in cluster:
                lineyLigands.append(ligand)
        _, _, _, identity = classifyLigands(ind)
        lineyLines = []
                
        for cluster in lineyClusters:            
            if len(cluster) >= minLineLength:
                G = networkx.Graph()
                for lineyLig in cluster:
                    G.add_node(lineyLig[0])                    
                for lineyLig in cluster:
                    for NN in lineyLig[1]['NNlist']:
                        if NN in lineyLigands:                        
                            G.add_edge(lineyLig[0], NN)                        

                dists = []
                for nodeA in G.nodes():
                    tmpdists = []
                    for nodeB in G.nodes():
                        if not sameLigands(nodeA, nodeB):
                            for pathl in [len(path) for path in networkx.all_simple_paths(G, nodeA, nodeB)]:
                                tmpdists.append(pathl)                        
                    dists.append(np.max(tmpdists))                                            
                lineyLines.append(np.max(dists))

        lineyLines = [line for line in lineyLines if line >= minLineLength]

        ind['lineylinetags'] = lineyLines

        return lineyLines

def lineyLineCount(ind, minLineLength=3):
    lineyLines = lineyLineTags(ind, minLineLength)

    return float(len(lineyLines))

def lineyLineSizeAVG(ind, minLineLength=3):
    lineyLines = lineyLineTags(ind, minLineLength)

    if lineyLines:
        return float(np.average(lineyLines))
    else:
        return 0.0

def lineyLineSizeMAX(ind, minLineLength=3):
    lineyLines = lineyLineTags(ind, minLineLength)

    if lineyLines:
        return float(np.max(lineyLines))
    else:
        return 0.0

def lineyLineSizeMIN(ind, minLineLength=3):
    lineyLines = lineyLineTags(ind, minLineLength)

    if lineyLines:
        return float(np.min(lineyLines))
    else:
        return 0.0

def lineyChainTags(ind, minChainLength):
    lineyClusters = groupLineyLigands(ind)    
    lineyChains = []

    for cluster in lineyClusters:
        if len(cluster) >= minChainLength:            
            lineyChains.append(cluster)

    return lineyChains    

def lineyChainCount(ind, minChainLength=3):
    lineyChains = lineyChainTags(ind, minChainLength)

    return float(len(lineyChains))

def lineyChainSizeAVG(ind, minChainLength=3):
    lineyChains = lineyChainTags(ind, minChainLength)

    if lineyChains:
        return float(np.average([len(chain) for chain in lineyChains]))
    else:
        return 0.0

def lineyChainSizeMAX(ind, minChainLength=3):
    lineyChains = lineyChainTags(ind, minChainLength)

    if lineyChains:
        return float(np.max([len(chain) for chain in lineyChains]))
    else:
        return 0.0

def lineyChainSizeMIN(ind, minChainLength=3):
    lineyChains = lineyChainTags(ind, minChainLength)

    if lineyChains:
        return float(np.min([len(chain) for chain in lineyChains]))
    else:
        return 0.0

def geneticDiversity(inds):
    genomes = {}    
    for ind in inds:
        genome = tuple(ind['genome'])
        if genome not in genomes:
            genomes[genome] = 1
        else:
            genomes[genome] += 1

    if len(inds) != np.sum(genomes.values()):        
        raise ValueError
    return genomes

def genomeConvergence(inds):
    genomes = geneticDiversity(inds)
    return 1.0 - float(len(genomes.keys()))/float(len(inds))  

def plotScanGen(scanData, scanLabel, scanIndices, interest, indexOffset, aggregateMode, plotName, cmap, fmt='.2g', vmin = None, vmax = None, annotate=False, silent=True, visual=True, dump=False, backup=False, dumpdir='plots'):    
    if not os.path.exists(dumpdir):
        os.mkdir(dumpdir)

    scanPlotIndex, scanPlotData = scanGen(scanData, interest, indexOffset, aggregateMode, silent=silent)

    if backup:
        pickle.dump(scanPlotIndex, open("{}-gen-scanPlotIndex.pickle".format(plotName), "wb"))
        pickle.dump(scanPlotData, open("{}-gen-scanPlotData.pickle".format(plotName), "wb"))

    plotData = np.zeros((len([i[1] for i in scanPlotData[0]]),len(scanPlotIndex)))
    annotData = np.zeros((len([i[1] for i in scanPlotData[0]]),len(scanPlotIndex)))

    cursorA = 0
    for i in range(len(scanPlotIndex)):
        scanPlotDatum = scanPlotData[cursorA]
        cursorB = 0
        for _ in [i[1] for i in scanPlotDatum]:        
            plotData[cursorB][cursorA] = scanPlotDatum[cursorB][1]
            annotData[cursorB][cursorA] = scanPlotDatum[cursorB][1]
            cursorB += 1
        cursorA += 1

    annot = annotData if annotate else False        

    ax = sns.heatmap(plotData, linewidth=1, annot=annot, cmap=cmap, vmin=vmin, vmax=vmax, fmt=fmt)
    plt.title('{}'.format(plotName))
    ax.set_xticks([i+0.5-indexOffset for i in scanIndices])
    ax.invert_yaxis()
    ax.set_xticklabels([i for i in scanIndices])
    plt.xlabel('{}'.format(scanLabel))
    plt.ylabel('generation')
    if dump:
        plt.savefig('{}/{}.png'.format(dumpdir,plotName));
    if visual:
        plt.show();

    
def plotScanCustom(scanData, scanLabel, scanIndices, interest, indexOffset, aggregateMode, plotName, cmap, interestKey, interestKeyLabel, tickerRange, tickerBlockSize, tickerBlockOffset, tickerInterval=0.0, fmt='.2g', vmin = None, vmax = None, annotate=False, linecolor='black', silent=True, visual=True, dump=False, backup=False, dumpdir='plots'):    
    if not os.path.exists(dumpdir):
        os.mkdir(dumpdir)

    scanPlotIndex, scanPlotData = scanCustom(scanData, interestKey, interestKeyLabel, tickerRange, tickerBlockSize, tickerBlockOffset, interest, indexOffset, aggregateMode, tickerInterval=tickerInterval,silent=silent)

    if backup:
        pickle.dump(scanPlotIndex, open("{}-{}-scanPlotIndex.pickle".format(plotName,interestKeyLabel), "wb"))
        pickle.dump(scanPlotData, open("{}-{}-scanPlotData.pickle".format(plotName,interestKeyLabel), "wb"))

    plotData = np.zeros((len([i[1] for i in scanPlotData[0]]),len(scanPlotIndex)))
    annotData = np.zeros((len([i[1] for i in scanPlotData[0]]),len(scanPlotIndex)))

    cursorA = 0
    for i in range(len(scanPlotIndex)):
        scanPlotDatum = scanPlotData[cursorA]
        cursorB = 0
        for _ in [i[1] for i in scanPlotDatum]:        
            plotData[cursorB][cursorA] = scanPlotDatum[cursorB][1]
            annotData[cursorB][cursorA] = scanPlotDatum[cursorB][1]
            cursorB += 1
        cursorA += 1

    annot = annotData if annotate else False        

    ax = sns.heatmap(plotData, linewidth=1, annot=annot, cmap=cmap, vmin=vmin, vmax=vmax, fmt=fmt, linecolor=linecolor)
    plt.title('{}'.format(plotName))
    ax.set_xticks([i+0.5-indexOffset for i in scanIndices])
    ax.set_xticklabels([i for i in scanIndices])
    ax.invert_yaxis()
    yticklabels = [0]
    for tickerBlock in range(tickerRange):
        yticklabels.append(tickerBlock*tickerBlockSize+tickerBlockOffset)
    ax.set_yticks([i for i in range(0,(tickerRange+1))])
    ax.set_yticklabels(yticklabels)
    ax.set_facecolor('#F5F5F5')
    plt.xlabel('{}'.format(scanLabel))
    plt.ylabel('{}'.format(interestKeyLabel))
    if dump:
        plt.savefig('{}/{}.png'.format(dumpdir,plotName));
    if visual:
        plt.show();

def measureLigandContact(xyzaFile, headersize=9, xyzallsize=2973, timestepinterval=100, intrange=1.8, silent=True):
    with open(xyzaFile) as f:
        content = f.readlines()
    content = [x.strip() for x in content] 

    steps = []
    timestep = 0
    while content:
        #remove header    
        xyz_iter = content[headersize:headersize+xyzallsize]
        steps.append((timestep, xyz_iter))
        content = content[headersize+xyzallsize:]
        timestep += timestepinterval

    contactData = []
    progress = 1
    for step in steps:
        if not silent:
            print('{}/{}'.format(progress, len(steps)))
        progress+=1
        time, coords = step
        coordsCLEANED = []
        for coord in coords:
            coordsCLEANED.append(coord.split())
        ligands = [i for i in coordsCLEANED if float(i[-1]) > 0.0]
        mems = [i for i in coordsCLEANED if float(i[1]) == 1.0]
        
        cLIG = []
        cMEM = []
        #convert strings to ints...    
        for ligand in ligands:
            ligand = [float(i) for i in ligand]
            cLIG.append(ligand)
        for mem in mems:
            mem = [float(i) for i in mem]
            cMEM.append(mem)
        ligands = cLIG
        mems = cMEM
        
        #count number of ligands that are within range
        ligandContact = 0
        for ligand in ligands:
            contact = False
            for mem in mems:
                if not contact:
                    lx, ly, lz = ligand[2], ligand[3], ligand[4]
                    mx, my, mz = mem[2], mem[3], mem[4]
                    dist = np.sqrt( (lx - mx)**2 + (ly - my)**2 + (lz - mz)**2 )
                    if dist <= intrange:
                        contact = True
            if contact:
                ligandContact += 1
        if ligandContact > len(ligands):
            raise ValueError
        contactData.append((time, ligandContact))

    return contactData, len(ligands)  

def smoothLigandContact(timeWindow, ligandContactWindow, windowSize, mode):
    window = np.ones(windowSize)/windowSize
    ligandContactWindowSmooth = np.convolve(ligandContactWindow, window, mode=mode)
    timeWindow = timeWindow[windowSize:][:-windowSize]
    ligandContactWindowSmooth = ligandContactWindowSmooth[windowSize:][:-windowSize]
    return timeWindow, ligandContactWindowSmooth

def jitterLigandContact(contactData, windowSize, mode):
    time = [i[0] for i in contactData]
    ligandContact = [i[1] for i in contactData]

    timeJitter = time[windowSize:][:-windowSize]
    ligandContactInterest = ligandContact[windowSize:][:-windowSize]
    _, ligandContactSmooth = smoothLigandContact(time, ligandContact, windowSize, mode)
    
    jitter = []
    for ligandContactInterest_i, ligandContactSmooth_i in zip(ligandContactInterest, ligandContactSmooth):
        jitter.append(ligandContactInterest_i - ligandContactSmooth_i)

    return timeJitter, jitter

def jitterLigandContactSUM(contactData, windowSize=10, mode='same', jitTolerance=0.1):
    _, jitter = jitterLigandContact(contactData, windowSize, mode)
    return np.sum([abs(jit) for jit in jitter if abs(jit) >= jitTolerance])

def jitterLigandContactMAX(contactData, windowSize=10, mode='same', jitTolerance=0.1):
    _, jitter = jitterLigandContact(contactData, windowSize, mode)
    return np.max([abs(jit) for jit in jitter if abs(jit) >= jitTolerance])

def digitalJitter(jitter, jitTolerance):
    jitDigital = []
    for jit in jitter:
        if abs(jit) >= jitTolerance:
            jitDigital.append(1)
        else:
            jitDigital.append(0)
    return jitDigital

def jitterLigandContactLIFETIME(contactData, windowSize=10, mode='same', jitTolerance=0.1):
    _, jitter = jitterLigandContact(contactData, windowSize, mode)
    jitter = digitalJitter(jitter, jitTolerance)

    totalLifeTime = 0    

    jitPacketALL = []
    jitPacket = []
    for jit in jitter:
        if jit == 1:
            jitPacket.append(1)
            totalLifeTime += 1
        elif jit == 0:
            if jitPacket:
                jitPacketALL.append(jitPacket)
                jitPacket = []
        else:
            raise ValueError
    jitPacketALL.append(jitPacket)

    averageLifeTime = np.average([len(jitPack) for jitPack in jitPacketALL])
    maxLifeTime = np.max([len(jitPack) for jitPack in jitPacketALL])
    minLifeTime = np.min([len(jitPack) for jitPack in jitPacketALL])

    return totalLifeTime, averageLifeTime, maxLifeTime, minLifeTime

def metastableLIFETIME(contactData):    
    ligandContact = [i[1] for i in contactData]

    prevLigandContact = 0
    digitalChange = []
    for nextLigandContact in ligandContact:
        if nextLigandContact != prevLigandContact:
            digitalChange.append(1)
            prevLigandContact = nextLigandContact
        else:
            digitalChange.append(0)

    totalLifeTime = 0

    stablePacketALL = []
    stablePacket = []
    for digitalChange_i in digitalChange:
        if digitalChange_i == 0:
            stablePacket.append(0)
            totalLifeTime += 1
        elif digitalChange_i == 1:
            if stablePacket:
                stablePacketALL.append(stablePacket)
                stablePacket = []
        else:
            raise ValueError
    stablePacketALL.append(stablePacket)

    averageLifeTime = np.average([len(stablePack) for stablePack in stablePacketALL])
    maxLifeTime = np.max([len(stablePack) for stablePack in stablePacketALL])
    minLifeTime = np.min([len(stablePack) for stablePack in stablePacketALL])    

    return totalLifeTime, averageLifeTime, maxLifeTime, minLifeTime

def cleanContactData(contactData, budTime):
    contactDataTRIMMED = []
    for contactData_i in contactData:
        if contactData_i[0] <= budTime:
            contactDataTRIMMED.append(contactData_i)
    return contactDataTRIMMED

def progressbar(it, prefix="", size=60):
    count = len(it) if len(it) > 0 else 1
    def _show(_i):
        x = int(size*_i/count)
        sys.stdout.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), _i, count))
        sys.stdout.flush()

    _show(0)
    for i, item in enumerate(it):
        yield item
        _show(i+1)
    sys.stdout.write("\n")
    sys.stdout.flush()

class UTF8Recoder:
    """
    Iterator that reads an encoded stream and reencodes the input to UTF-8
    """
    def __init__(self, f, encoding):
        self.reader = codecs.getreader(encoding)(f)

    def __iter__(self):
        return self

    def next(self):
        return self.reader.next().encode("utf-8")

class UnicodeReader:
    """
    A CSV reader which will iterate over lines in the CSV file "f",
    which is encoded in the given encoding.
    """

    def __init__(self, f, dialect=csv.excel, encoding="utf-8", **kwds):
        f = UTF8Recoder(f, encoding)
        self.reader = csv.reader(f, dialect=dialect, **kwds)

    def next(self):
        row = self.reader.next()
        return [unicode(s, "utf-8") for s in row]

    def __iter__(self):
        return self

class UnicodeWriter:
    """
    A CSV writer which will write rows to CSV file "f",
    which is encoded in the given encoding.
    """

    def __init__(self, f, dialect=csv.excel, encoding="utf-8", **kwds):
        # Redirect output to a queue
        self.queue = cStringIO.StringIO()
        self.writer = csv.writer(self.queue, dialect=dialect, **kwds)
        self.stream = f
        self.encoder = codecs.getincrementalencoder(encoding)()

    def writerow(self, row):
        self.writer.writerow([s.encode("utf-8") for s in row])
        # Fetch UTF-8 output from the queue ...
        data = self.queue.getvalue()
        data = data.decode("utf-8")
        # ... and reencode it into the target encoding
        data = self.encoder.encode(data)
        # write to the target stream
        self.stream.write(data)
        # empty queue
        self.queue.truncate(0)

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)

def crt2SphPol(cart):
    x = cart[0]
    y = cart[1]
    z = cart[2]
    r = math.sqrt(x*x+y*y+z*z)
    t = math.acos(z/(r))
    p = math.atan2(y,x)
    return(r,t,p)

def sphPol2Crt(sph):
    r = sph[0]
    t = sph[1]
    p = sph[2]
    x = r*math.sin(t)*math.cos(p)
    y = r*math.sin(t)*math.sin(p)
    z = r*math.cos(t)
    return(x,y,z)

def cartDistMag(a,b):
    return np.sum(map(lambda a,b: (a-b)**2,a,b))

def asPolars(sphere):
    polarSphere = []
    for v in sphere:
        polarSphere.append(crt2SphPol(v))
    return polarSphere


def coverSphere(radius,coords):
    newCoords = []
    for c in coords:
        cSph = crt2SphPol(c)
        cSphNew = (radius,cSph[1],cSph[2])
        newCoords.append(sphPol2Crt(cSphNew))
    return newCoords

def getMinimumRadius(coords):
    minD = cartDistMag(coords[0],coords[1])
    for i in range(len(coords)):
        for j in range(i+1,len(coords)):
            c1 = coords[i]
            c2 = coords[j]
            d = cartDistMag(c1,c2)
            if d < minD:
                minD = d
    return minD

mag = lambda v : np.sqrt(np.sum([i*i for i in v]))

def ang3D(v1,v2):
    return np.arctan2(mag(np.cross(v1,v2)), np.dot(v1,v2))

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)



def frange(start, stop, step):
    x = start
    while x < stop:
        yield x
        x += step

def readXYZA(filepath,headerSize=9):
    data = {}
    with open(filepath, "r") as f:
        lines = f.readlines()
        ts = 0
        steps = []
        hPos = 0
        currentStep = {}
        data['atoms'] = []
        idIndex = -1
        typeIndex = -1
        storedAtomTypes = False
        for i in range(len(lines)):
            if str('ITEM: TIMESTEP') in lines[i]:
                if 't' in currentStep:
                    steps.append(currentStep)
                    storedAtomTypes = True   
                hPos = i
                ts = int(lines[i+1])
                currentStep = {'t':ts,'data':[]}
                
            elif str('ITEM: NUMBER OF ATOMS') in lines[i]:
                data['natoms'] = int(lines[i+1])  
            elif str('ITEM: BOX BOUNDS') in lines[i]:
                currentStep['xsize'] = [float(b) for b in lines[i+1].split(' ')]
                currentStep['ysize'] = [float(b) for b in lines[i+2].split(' ')]
                currentStep['zsize'] = [float(b) for b in lines[i+3].split(' ')]
            elif str('ITEM: ATOMS') in lines[i]:
                data['columns'] = lines[i].replace('\n','').split(' ')[2:]
                if not storedAtomTypes:
                    if 'id' in data['columns']:
                        idIndex = data['columns'].index('id')
                    if 'type' in data['columns']:
                        typeIndex = data['columns'].index('type')
            if(i-hPos>headerSize-1):
                atomStep = {}
                atomData = lines[i].replace('\n','').split(' ')
                if len(atomData) != len(data['columns']):
                    continue
                if not storedAtomTypes:
                    if idIndex>=0 and typeIndex>=0:
                        data['atoms'].append((int(atomData[idIndex]),int(atomData[typeIndex])))
                for j,c in enumerate(data['columns']):
                    atomStep[c] = float(atomData[j]) if ('.' in atomData[j] or 'e' in atomData[j]) else int(atomData[j])
                currentStep['data'].append(atomStep)
        data['steps'] = steps
    return data

def dataToDataframe(data,filepath):
    cols = ['timestep']
    for c in data['columns']:
        cols.append(c)

    dfdata = []
    for s in data['steps']:
        for a in s['data']:
            td = []
            td.append(s['t'])
            for j in data['columns']:
                td.append(a[j])
            dfdata.append(tuple(td))

    df = pd.DataFrame(dfdata, columns = cols) 
    df.to_csv(filepath.replace('xyza','csv'),index=False)
    return df

def xyzaToDataframe(filepath,headerSize=9):
    d = readXYZA(filepath,headerSize=9)
    df = dataToDataframe(d,filepath)
    return df

def getBuddingTime(data):
    #this takes an input from readxyza, not a filepath
    if not 'c_cls' in data['columns']:
        return -1
    for s in data['steps']:
        clus = []
        for a in s['data']:
            clus.append(a['c_cls'])
        cSet = set(clus)
        clusSizes = {}
        for c in cSet:
            clusSizes[c] = clus.count(c)
        largeClusters = 0
        for k,v in clusSizes.iteritems():
            if v > 25:
                largeClusters += 1
        if largeClusters > 1:
            return s['t']
    return -1

def pointCloudToDataFrame(data):
    #turns a series of xyz points into a nice pandas dataframe
    df = pd.DataFrame(data,columns=['x','y','z'])
    return df

def pointCloudToDistanceMatrix(data):
    #gets pairwise distances of a cloud of xyz points
    df = pointCloudToDataFrame(data)
    dists = distance_matrix(df.values, df.values)
    return dists

def pointCloudToInverseDistanceMatrix(data):
    #gets inverse pairwise distances of a cloud of xyz points
    df = pointCloudToDataFrame(data)
    distdf = pd.DataFrame(distance_matrix(df.values, df.values), index=df.index, columns=df.index)
    distdf = distdf.apply(lambda a : [1.0/float(b) if b > 0 else 0.0 for b in a],axis=0)
    dists = distdf.values
    return dists

def removeSelfLoops(G):
    #removes the self edges from a graph
    G.remove_edges_from(G.selfloop_edges())
    return G

def pointCloudToDistanceGraph(data, removeSelfEdges=True, inverse=False):
    #the full package - turns a cloud of xyz points into a networkx graph with distances scaled by linear distance
    dists = pointCloudToDistanceMatrix(data) if inverse else pointCloudToInverseDistanceMatrix(data)
    dt = [('weight',float)]
    A=np.matrix(dists,dtype=dt)
    G = networkx.from_numpy_matrix(A)
    if removeSelfEdges:
        G = removeSelfLoops(G)
    return G

def orthoProjection(data,scale=1.0,offset=0.0):
    #turns 3D data into an orthographic projected version, useful!
    ortho = []
    for d in data:
        b = []
        b.append(scale*d[0] + offset)
        b.append(scale*d[2] + offset)
        ortho.append(b)
    return ortho

def pointCloudToPositionDictionary(data):
    #makes a position dictionary in the form expected by networkx
    posDict = {}
    for i,d in enumerate(orthoProjection(data)):
        posDict[i] = d
    return posDict

def getRotationDataFromXYZA(simData):
    coreId = -1
    refId = -1
    for i,a in enumerate(simData['steps'][0]['data']):
        if a['type'] == 2:
            coreId = i
        if a['type'] == 3:
            refId = i

    r = []
    rotation = []
    
    for i,s in enumerate(simData['steps']):
        core = s['data'][coreId]
        ref = s['data'][refId]
        vCore = [core['x'],core['y'],core['z']]
        vRef = [ref['x'],ref['y'],ref['z']]
        r.append(np.subtract(vRef,vCore))
        if i>0:
            rotation.append(r[-1])
            
    return rotation

def getRotationData(d):
    coreTraj=d[d.type==2].sort_values(by=['timestep'])[['x','y','z']].values
    refTraj=d[d.type==3].sort_values(by=['timestep'])[['x','y','z']].values
    rTraj = [np.subtract(refTraj[i],coreTraj[i]) for i,v in enumerate(coreTraj)]
    return rTraj

def getPolarRotationData(simData):
    rotation = getRotationData(simData)
    return [crt2SphPol(r) for r in rotation]

def getAngularDisplacement(simData):
    rotation = getPolarRotationData(simData)
    return [np.subtract(r,rotation[0]) for r in rotation]

def getDisplacement(simData):
    rotation = getRotationData(simData)
    return [np.subtract(r,rotation[0]) for r in rotation]

def plotRotationTrajectory(data):
    colors = []
    colStep = 1.0/float(len(data))
    for i in range(len(data)):
        colors.append((colStep*i,0.0,1.0-colStep*i,1.0))
        
    startPoint = 0
    X, Y, Z = zip(*data[startPoint:])
    fig = plt.figure(figsize=(13, 13))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X,Y,Z,alpha=0.4)
    ax.scatter(X, Y, Z,c=colors[startPoint:])
    return fig

def plotRotationTrajectoryOnSphere(data,fig=plt.figure(figsize=(13, 13))):
    colors = []
    colStep = 1.0/float(len(data))
    for i in range(len(data)):
        colors.append((colStep*i,0.0,1.0-colStep*i,1.0))
        
    startPoint = 0
    X, Y, Z = zip(*data[startPoint:])
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X,Y,Z,alpha=0.4)
    ax.set_xlim([-4.0,4.0])
    ax.set_ylim([-4.0,4.0])
    ax.set_zlim([-4.0,4.0])
    u, v = np.mgrid[0:2*np.pi:200j, 0:np.pi:200j]
    sx = 4.0*np.cos(u)*np.sin(v)
    sy = 4.0*np.sin(u)*np.sin(v)
    sz = 4.0*np.cos(v)
    ax.plot_wireframe(sx, sy, sz, color="r",alpha=0.05)
    ax.scatter(X, Y, Z,c=colors[startPoint:])
    return fig

def getCorr(trajvectors,delta):
    trajcorr = [np.dot(t,trajvectors[i+delta]) for i,t in enumerate(trajvectors[1:len(trajvectors)-delta-1])]
    corr = np.mean(trajcorr)
    return corr

def getCorrs(traj,deltas):
    corrs = []
    trajvectors = [t-traj[i] for i,t in enumerate(traj[1:])]
    for d in range(deltas):
        corr = getCorr(trajvectors,d)
        corrs.append(corr)
    return corrs

def generateRotationSummary(traj,deltas):
    summ = {}
    corrs = getCorrs(traj,deltas)
    trajdisplacement = [np.linalg.norm(t-traj[0]) for i,t in enumerate(traj[1:])]
    trajmsd = [t*t for t in trajdisplacement]
    
    G = pointCloudToDistanceGraph(traj)

    #REMOVE THIS NEXT TIME YOU LOAD THE KERNEL!
    # for u,v,d in G.edges(data=True):
    #     print d
        
    G=pruneNetwork(G,2.0,pruneNodes=False)

    den = networkx.density(G)
    clus = 0.0
    if den > 0.0:
        clus = networkx.average_clustering(G)

    
    summ['corrs'] = corrs
    summ['msd'] = trajmsd
    summ['graph'] = G
    summ['density'] = den
    summ['clustering'] = clus
    return summ



def generateSummaries(xyzapath):
    xyzas = [os.path.join(xyzapath,f) for f in filter(lambda x: '.xyza' in x, os.listdir(xyzapath))]
    summ = {}
    n = len(xyzas)

    for i,x in enumerate(xyzas):
        print(str(x) + " file " + str(i+1) + " of " + str(n))
        summ[x] = {}
        d = xyzaToDataframe(x)
        traj = getRotationData(d)
        s = generateRotationSummary(traj,5)
        bt = getBuddingTime(d)
        summ[x] = s
        summ[x]['bt'] = bt

    return summ
    
