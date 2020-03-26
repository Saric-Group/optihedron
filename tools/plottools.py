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
import analysistools as atools

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
