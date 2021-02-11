import numpy as np
import laspy as lp
import scipy
from laspy.file import File
from scipy.spatial.ckdtree import cKDTree as KDTree
import time 
import datetime


class Scan():
	def __init__(self,filepath,skipinterval=1,buildTree=True):

		self.filepath = filepath.filepath
		self.file = File(self.filepath,mode="r")
		self.scale = self.file.header.scale[0]
		self.offset = self.file.header.offset[0]
		if buildTree:
			self.tree = KDTree(np.vstack([self.file.x[::skipinterval], self.file.y[::skipinterval], self.file.z[::skipinterval]]).transpose())
			self.treeexis = True			
		
		self.datetime = filepath.datetime

	def knear(self,point,k):
		if not self.treeexis:
			raise ValueError("Tree Is Not Built")
		return self.tree.data[self.tree.query(point,k=k)[1]]

	def radialcluster(self,point,radius):
		if not self.treeexis:
			raise ValueError("Tree Is Not Built")
		neighbor = self.tree.data[self.tree.query(point,k=1)[1]]
		points = self.tree.data[self.tree.query_ball_point(neighbor,radius)]
		return np.array(points)

	def pointSet(self):
		return np.vstack([self.file.x[::skipinterval], self.file.y[::skipinterval], self.file.z[::skipinterval]])

class Particleset():  
	def __init__(self,points=None,center=None,weight=None):
		self.points = points
		self.center = center
		self.weight = weight
		self.normalize()
		self.gen_covariance()

	def normalize(self):
		if self.center is not None:
			self.points[:,:2] -= np.mean(self.points[:,:2]) # normalize by either center or mean (TBD)

	def set_weight(self,weight):
		self.weight = weight

	def gen_covariance(self):
		if self.points is not None:
			self.cov = np.cov(self.points.T)
