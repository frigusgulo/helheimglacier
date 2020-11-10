# create training dataset for GP kernel learning

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


class dataset(torch.utils.data.Dataset):
	def __init__(self, points,elevs,transform=None):
		self.clusters = zip(points,elevs)

	def __len__(self):
		return(self.clusters.len())
	
	def __getitem__(self,idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		return self.clusters[idx,:]





pointa = np.array([526807,7365496]).astype(np.int)
pointb = np.array([534835,7359581]).astype(np.int)

easting = np.linspace(pointa[0],pointb[0], np.abs(np.int((pointa[0]-pointb[0])/120)))
northing = np.linspace(pointa[1],pointb[1],np.abs(np.int((pointa[1]-pointb[1])/120)))

points = [(x,y) for x in easting for y in northing]

file = "/home/dunbar/Research/helheim/data/160802_000217.laz" 
scan = scan(file)

pointset = []
for point in points:
	point = point +  (500,) # add elevation value so nearest point on the plane is found
	cluster = scan.radialcluster(point,100)
	#print("Cluster Shape: {} \n".format(cluster.shape))
	if cluster.shape[0] > 100:
		pointset.append(cluster)
print(len(pointset))
dataset = np.array(pointset)
np.save('/home/dunbar/Research/helheim/data/pcluster',dataset)
