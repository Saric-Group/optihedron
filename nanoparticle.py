from ga import phenome
from ga import grayencoder as ge
from tools import listtools as lt
import numpy as np
import math

class Ligand:
	def __init__(self,eps,sig,rad,polAng,aziAng,mass=1,cutoff=2.0):
		self.rad = rad
		self.polAng = polAng
		self.aziAng = aziAng
		self.eps = eps
		self.sig = sig
		self.mass = mass
		self.cutoff = cutoff
		self.size = 1.0

	def __str__(self):
		ligstr= "rad:"+str(self.rad)+", polAng:"+str(self.polAng)+", aziAng:"+str(self.aziAng)+", eps:"+str(self.eps)
		ligstr+= ", sig:"+str(self.sig)+", mass:"+str(self.mass)+", cutoff:"+str(self.cutoff)
		return ligstr

class NanoParticle:
	ligands = []
	def __init__(
		self,
		x=0.0,
		y=0.0,
		z=0.0,
		mass=1.0,
		eps=1.0,
		sig=4.0,
		cutoff=2.0**(1.0/6.0)
		):
		self.x = x
		self.y = y
		self.z = z
		self.mass = mass
		self.eps = eps
		self.sig = sig
		self.cutoff = cutoff
		self.ligands=[]

	def addLigand(self,ligand):
		if(isinstance(ligand,Ligand)):
			self.ligands.append(ligand)

	def spaceIsOccupied(self,polarTargetAngle,azimuthalTargetAngle):
		target_v = [
		self.sig*math.sin(polarTargetAngle)*math.cos(azimuthalTargetAngle),
		self.sig*math.sin(polarTargetAngle)*math.sin(azimuthalTargetAngle),
		self.sig*math.cos(polarTargetAngle)
		]
		for ligand in self.ligands:
			ligand_v = [
			ligand.rad*math.sin(ligand.polAng)*math.cos(ligand.aziAng),
			ligand.rad*math.sin(ligand.polAng)*math.sin(ligand.aziAng),
			ligand.rad*math.cos(ligand.polAng)
			]

			d = np.linalg.norm(np.subtract(ligand_v,target_v))
			if d<ligand.size:
				return True
		return False

	def __str__(self):
		protstr = "x:"+str(self.x)+", y:"+str(self.y)+", m:"+str(self.mass)
		protstr += ", eps:"+str(self.eps)+", sig:"+str(self.sig)+", cut:"+str(self.cutoff)
		protstr += "\n"+str(len(self.ligands))+" ligands"
		i=0
		for l in self.ligands:
			i+=1
			protstr += "\nligand " + str(i) +" - "+ str(l)
		return protstr

class NanoParticlePhenome(phenome.Phenome):

	def __init__(
		self,
		ind,
		exprPlaces,
		epsPlaces,
		polarAngPlaces,
		azimuthalAngPlaces,
		minEps,
		maxEps
		):
		self.exprPlaces = exprPlaces
		self.epsPlaces = epsPlaces
		self.polarAngPlaces = polarAngPlaces
		self.azimuthalAngPlaces = azimuthalAngPlaces
		self.minEps = minEps
		self.maxEps = maxEps
		phenome.Phenome.__init__(self,ind)

	def constructGenome(self,ind):
		geneSize = self.exprPlaces+self.epsPlaces+self.polarAngPlaces+self.azimuthalAngPlaces
		if len(ind) < geneSize:
			print("genome too short to construct phenome")
			return None
		genelist = lt.subdivide(list(ind),geneSize)
		if len(genelist[-1]) < geneSize:
			genelist = genelist[:-1]
		genes = []
		for g in genelist:
			gene = {}
			exprGene = g[0:self.exprPlaces]			
			epsGene = g[self.exprPlaces:self.exprPlaces+self.epsPlaces]
			polarAngGene = g[self.exprPlaces+self.epsPlaces:self.exprPlaces+self.epsPlaces+self.polarAngPlaces]
			azimuthalAngGene = g[self.exprPlaces+self.epsPlaces+self.polarAngPlaces:self.exprPlaces+self.epsPlaces+self.polarAngPlaces+self.azimuthalAngPlaces]
			gene['expr'] = True if ge.read(exprGene)/ge.max(exprGene) >= 0.5 else False 			
			gene['eps'] = (ge.read(epsGene)*(self.maxEps-self.minEps))/ge.max(epsGene)
			gene['polAng'] = (ge.read(polarAngGene)*(3.141)/ge.max(polarAngGene))
			gene['aziAng'] = (ge.read(azimuthalAngGene)*(6.2831)/ge.max(azimuthalAngGene))
			genes.append(gene)
		return genes


	def constructPhenome(self,ind):
		self.particle = NanoParticle()
		for g in self.genome:
			if g['expr']:				
				if not self.particle.spaceIsOccupied(g['polAng'],g['aziAng']):
					self.particle.addLigand(Ligand(g['eps'],1,4,g['polAng'],g['aziAng']))			
		return self.particle