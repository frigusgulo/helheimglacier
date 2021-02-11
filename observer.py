# Observer class

import datetime
import numpy as np 
from .scan import Scan 

class Scanset:
	'''
	A sequence of LiDAR scan observations

	Attributes:
		scans (list) : list of filepaths to LiDAR point clouds stored in scrictly increasing time

		density (int) : skip Interval of points to store in the KD tree

		load (bool): Weather to load all scans into memory or untill queried 
	

	'''

	def __init__(
		self,
		scans: list,
		density: int=2,
		load: bool = False):


	if len(scans) < 2:
		raise ValueError("Images are not two or greater")
    if any(scan.datetime is None for scan in self.scans):
		raise ValueError(f"Image {i} is missing datetime")
	if load:
		self.scans = [Scan(filepath.filepath,skipinterval=self.density) for filepath in self.scans]

	self.scans = sorted(key = lambda i: scan.datetime, self.scans)
	self.date_times = [scan.datetime for scan in self.scans]
	time_deltas = np.array([dt.total_seconds() for dt in np.diff(self.date_times)])
	if any(time_deltas <= 0):
		raise ValueError("Image datetimes are not stricly increasing")
	self.datetimes = np.array(self.date_times)

	def index(self,index) -> Scan:
		if self.load:
			return self.scans[index]

		filepath = self.scans[index]


		return Scan(filepath.filepath,skipinterval=self.density)