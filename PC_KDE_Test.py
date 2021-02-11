# Perform classification tests on reference vs hypothesis clusters

import numpy as np
import laspy as lp
import numpy as np
import scipy
from laspy.file import File
from scipy.spatial.kdtree import KDTree
import torch 
import pickle
import time as time
import matplotlib.pyplot as plt
import scipy.stats
from numpy.random import (normal,uniform)
import matplotlib.cm as cm
import itertools
from matplotlib.patches import Circle
from glob import glob
from os.path import join
import itertools
import inspect


class scan():
    def __init__(self,filepath):
        start = time.time()
        self.file = File(filepath,mode="r")
        self.scale = self.file.header.scale[0]
        self.offset = self.file.header.offset[0]
        self.tree = KDTree(np.vstack([self.file.x, self.file.y, self.file.z]).transpose())
        self.time = self.file.header.get_date()
        end = time.time() - start
        print("Time Elapsed: {}".format(end))

    def nearNeighbor(self,point,k=1):
        return self.tree.query(point,k=k)
    
    def NNN(self,point,k):
        return self.tree.data[self.tree.query(point,k=k)[1]]
    
    def radialcluster(self,point,radius):
        neighbor = self.tree.data[self.tree.query(point,k=1)[1]]
        points = self.tree.data[self.tree.query_ball_point(neighbor,radius)]
        return np.array(points)


class Particleset():  
    
    def __init__(self,points=None,center=None,label=None,weight=None):
        self.points = points
        self.center = center
        self.weight = weight
        self.gen_covariance()
        self.testlabel = np.zeros((1,100))
        self.label = label
    def set_weight(self,weight):
        self.weight = weight

    def settestlabel(self,weight,index):
    	self.testlabel[:,index] = weight

    def gen_covariance(self):
        if self.points is not None:
            self.cov = np.cov(m=self.points.T)
    
class Observer(Particleset):
    def __init__(self,obs):
        self.observations = obs
        self.observations.sort(key= lambda scan: scan.time)
        #self.particles = None

        
    def kernel_function(self,test,ref):
        # Evaluate the liklihood of one KDE under another
        mean_test = test.points
        cov_test = test.cov
        mean_ref = ref.points
        cov_ref = ref.cov
        assert mean_test.shape == mean_ref.shape
        gamma = mean_test - mean_ref
        sigma = cov_test + cov_ref
        A = 1/((2*np.pi**(3/2))*(np.linalg.det(sigma)**(.5)))
        B = np.exp((-.5)*gamma@np.linalg.inv(sigma)@gamma.T)
        C = 1/(np.max(mean_test.shape))
        return (C**2)*np.sum(A*B)





'''
Experiment:

1) Get 10 Clusters on a grid from a scan

2) For each cluster, create 10 copies and perterb the points with random noise while assigning the 
cluster a label

3) Iterate through all test clusters and see if the highest "weight" is assigned to the true cluster



Doug,
I have come up with something of an experiment that evaluates the accuracy of this approach to "labeling" hypothesis clusters.
This was inspired predominately by what I have read in https://iis.uibk.ac.at/public/papers/Xiong-2013-3DV.pdf (Section 3.1-3.3)

 I formulated my approach as follows:

Given a cluster of reference points
and a set of test clusters:

1) On a grid of "centers" in the UTM coordinate system, retrieve from the KD tree  a cluster of 100 points that are Nearest Neighbors to the center 
and assign a respective label to each cluster [0-99]

2) Create 10 "test" clusters from each labeled cluster: Perturb these points with white gaussian noise and assign their true label to that of the labeled cluster

3) Iterate through all the perturbed "test" clusters and compute the expected likelihood between them and each labeled cluster, assigning the lilihood of each labeled
cluster to the "testlabel" array

4) Compute accuracy by the number of times the largest computed likelihood corresponds to the true label of the cluster


'''
files = glob(join("/home/dunbar/Research/helheim/data/misc/lazfiles","*.laz"))
refscan = scan(files[0])
m = 3
n = 10
easting = np.linspace(534207.30,534407.30,m)
northing = np.linspace(7361840.57,7362040.57,m)
grid = [np.array([x,y,1000]) for x in easting for y in northing]
labeled_scans = [Particleset(points=refscan.NNN(point,k=300),label=i) for i,point in enumerate(grid)] # Step 1

def perturb_points(points):
	n,m = points.shape
	new = points.copy()
	for i in range(m):
		noise = np.random.normal(0,.05,n)
		new[:,i] += noise.T

	return new

testscans = []
for scan in labeled_scans:
	points = scan.points
	for i in range(n):
		set_ = perturb_points(points)
		testscans.append(Particleset(points=set_,label=scan.label)) # Step 2

observer = Observer([refscan])

for testscan in testscans:
	for labelscan in labeled_scans:
		weight = observer.kernel_function(testscan,labelscan) # Step 3
		testscan.settestlabel(weight,labelscan.label)

count = 0
for testscan in testscans:
	
	if int(np.argmax(testscan.testlabel)) == int(testscan.label): # Step 4
		count += 1
	#else:
	#	print(np.searchsorted(testscan.testlabel)[:4], testscan.label)
print("Classification Accuracy: {}".format(count/(m**2*n)))

