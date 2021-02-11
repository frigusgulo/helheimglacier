 
import matplotlib
matplotlib.use('agg')
import glimpse
import sys
import datetime
import numpy as np
import os
import glob
from lmfit import Parameter

SNAP = datetime.timedelta(hours=1)
MAXDT = datetime.timedelta(days=1)
MATCH_SEQ = (np.arange(3) + 1)
MAX_RATIO = 0.6
MAX_ERROR = 0.03 # fraction of image width
N_MATCHES = 50
MIN_MATCHES = 50
#root = '~/Research/wolverine/data'
TOUNGE_MATCHES_PATH = "~/Research/helheim/matches.pkl"
class optView():
	def __init__(self,savedir,imagedir,basemodelpath,basedict,imgpoints,worldpoints,mask):
		self.imagepaths =  glob.glob(os.path.join(imagedir,'*.jpg'),recursive=True)
		self.basecam = glimpse.Camera.from_json(basemodelpath)
		self.imagepoints = imgpoints
		self.worldpoints = worldpoints
		self.refimg = glimpse.Image(path=self.imagepaths[0],exif=glimpse.Exif(self.imagepaths[0]),cam= self.basecam)
		self.savedir = savedir
		print(self.basecam)
		self.mask = np.load(mask)

	def modelRef(self):
		points = glimpse.optimize.Points(self.refimg.cam,self.imagepoints,self.worldpoints)
		camera = glimpse.optimize.Cameras(cams=[self.refimg.cam],controls=[points],cam_params={'f':([0,1],[1000,1000],[1794,1347]),'viewdir':([0,1,2],[-50,-90,-20],[50,90,20]),'xyz':([0,1,2],[536910.8039, 7357000.081, 400],[536955.8039, 7357042.081, 500])})
		fit = camera.fit(method='lm')
		camera.set_cameras(fit)


	def getImages(self):
		self.images = []
		[self.images.append( glimpse.Image(path=imagepath,exif=glimpse.Exif(imagepath),cam=self.refimg.cam.copy()) ) for imagepath in self.imagepaths[1:]]
		self.images.sort(key= lambda img: img.datetime)
		print("Found {} Images \n".format(len(self.images)))

	def iterMatch(self,setSize=15):
		subSet = []
		overlap = []
	
		for i,image in enumerate(self.images):
			subSet.append(image)
			if i > 0 and i % setSize == 0 or i == len(self.images)-1:
		
				observercams = glimpse.optimize.ObserverCameras(
			    	observer = glimpse.Observer(subSet),
			    	anchors=[0]
			    	)
				observercams.build_keypoints()
				observercams.build_matches()
				print("\nFitting\n")
				fit = observercams.fit()#method='Nelder-Mead',full=True,tol=1e-8)#,options={"maxiter":100,"disp":True})
				observercameras.set_cameras(fit.params)
	
				subSet = []
				self.saveCams(observercams)
	
				
                              

	def run(self):
		self.modelRef()
		self.getImages()
		self.iterMatch()

	def saveCams(self,observercams):
		images = [image for image in observer.images for observer in [observercams.observer]]
		for image in images:
			filename = images.path.replace("jpg","JSON")
			print("\n Writing {} \n".format(filename))
			image.cam.to_json(path=path)

if __name__ == "__main__":


	# DIDNT ENTER IMAGE CONTROL POINTS CORRECTLY
	
	dual2worldpts = np.array([[537464,7379214,1096],[528472.74,7370568.6,416],[537870.25,7365624.42,167],[535469.85,7364576,237]])
	dual2imgpts = np.array([[165,686],[228,38],[248,772],[240,515]])[:,::-1]
	dual2basedict = dict(sensorsz=(5.70,4.28),xyz=(536935.8039, 7357022.081, 469.526), viewdir=(0,0,0))
	dual2savedir = "/home/dunbar/Research/helheim"
	dual2imgdir = "/home/dunbar/Research/helheim/data/observations/stardot2"
	dual2mask = "/home/dunbar/Research/helheim/data/misc/stardot2mask.npy"
	


	dual1worldpts = np.array([[537443.79,7379236.59,1086],[541582,7362810,110],[540547,7362878,119],[542619.85,7362388.83,111]])
	dual1imgpts = np.array([[155,59],[278,783],[261,671],[286,926]])[:,::-1]
	dual1basedict = dict(sensorsz=(5.70,4.28),xyz=(536935.8039, 7357022.081, 469.526), viewdir=(0,0,0))
	dual1savedir = "/home/dunbar/Research/helheim"
	dual1imgdir =  "/home/dunbar/Research/helheim/data/observations/stardot1"
	dual1mask = '/home/dunbar/Research/helheim/data/misc/dual1mask.npy'

	
	basemodelpath = "/home/dunbar/Research/helheim/data/intrinsicmodel.json"
	
	dual2viewopt = optView(dual2savedir,dual2imgdir,basemodelpath,dual2basedict,dual2imgpts,dual2worldpts,mask=dual2mask)
	dual2viewopt.run()
	
	
	dual1viewopt = optView(dual1savedir,dual1imgdir,basemodelpath,dual1basedict,dual1imgpts,dual1worldpts,mask=dual1mask)
	dual1viewopt.run()
