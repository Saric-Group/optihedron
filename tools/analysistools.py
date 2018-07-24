import numpy as np
import copy
import time

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
	if ligandA.__dict__ == ligandB.__dict__:
		return True
	else:
		return False

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
            for ligand in ligandsTmp:
                NNlist = ligand[1]['NNlist']
                existingCopies = 0
                for NN in NNlist:
                    if sameLigands(seed[0], NN):
                        existingCopies += 1
                if existingCopies == 1:                                    
                    nextSeedQueue.append(ligand)
                    clusterTmp.append(ligand)
                    ligandsTmp.remove(ligand)
                elif existingCopies > 1:
                    raise ValueError
        clusters.append(clusterTmp)        

    if not silent:
        print('ligands: {}'.format(len(ligands)))
        print('clusters: {}'.format(len(clusters)))
        print('clustered in {}s'.format(time.time()-startTime))

    return clusters

def groupLineyLigands(ind):
    if 'lineyTags' in ind:
        return ind['lineyTags']
    else:
        identity = ind['ligtags'][3]
        lineyLigands = []

        for ligand, ligandTags in identity.items():
            if ligandInfo['character'] == 'liney':
                lineyLigands.append((ligand,ligandTags)) 
        lineyClusters = clusterLineyLigands(lineyLigands)
        ind['lineyTags'] = lineyClusters
        return lineyClusters