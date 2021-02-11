import laspy as lp
import numpy as np
import scipy
from laspy.file import File
import glob
from scipy.spatial.kdtree import KDTree
import torch 
import pickle
import time as time


class cluster():
	def __init__(self, points):
		self.points = np.array(points)
		print(self.points.shape)
		self.roughness = np.nanstd(self.points[:,-1])

	def boundpoints(self):
		x_min = np.min(self.points[:,0])
		x_max = np.max(self.points[:,0])
		y_min = np.min(self.points[:,1])
		y_max = np.max(self.points[:,1])
		return np.array([[x_min,x_max],[y_min,y_max]])


class scan(cluster):
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

	def radialcluster(self,point,radius):
		neighbor = self.tree.data[self.tree.query(point,k=1)[1]]
		points = self.tree.data[self.tree.query_ball_point(neighbor,radius)]
		print("{} Points \n".format(points.shape[0]))
		return np.array(points)
