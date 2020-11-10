import matplotlib
matplotlib.use('agg')
import glimpse
from glimpse.imports import (sys, datetime, matplotlib, np, os)
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
		self.basecam = glimpse.helpers.merge_dicts(glimpse.Camera.read(basemodelpath).as_dict(),basedict)
		self.imagepoints = imgpoints
		self.worldpoints = worldpoints
		self.refimg = glimpse.Image(path=self.imagepaths[0],exif=glimpse.Exif(self.imagepaths[0]),cam=self.basecam.copy())
		self.savedir = savedir
		print(self.basecam)
		self.mask = np.load(mask)

	def modelRef(self):
		points = glimpse.optimize.Points(self.refimg.cam,self.imagepoints,self.worldpoints)
		camera = glimpse.optimize.Cameras(cams=[self.refimg.cam],controls=[points],cam_params={'f':([0,1],[1077,606],[1347,1794]),'viewdir':([0,1,2],[-40,-45,-20],[45,45,20]),'xyz':([0,1,2],[536915.8039, 7357002.081, 449.526],[536955.8039, 7357042.081, 489.526])})
		fit = camera.fit()#(method='lm')
		camera.set_cameras(fit)


	def getImages(self):
		self.images = []
		[self.images.append( glimpse.Image(path=imagepath,exif=glimpse.Exif(imagepath),cam=self.refimg.cam.copy()) ) for imagepath in self.imagepaths[1:]]
		self.images.sort(key= lambda img: img.datetime)
		print("Found {} Images \n".format(len(self.images)))

	def iterMatch(self,setSize=100):
		subSet = []
		overlap = []
	
		for i,image in enumerate(self.images):
			subSet.append(image)
			if i > 0 and i % setSize == 0 or i == len(self.images)-1:
				try:
					subSet = overlap + subSet
				except:
					print("overlap empty")
				print("\nSubset: {} Images: {}\n".format((i+1),len(subSet)))
				matcher = glimpse.optimize.KeypointMatcher(subSet)
				mask = [self.mask for _ in subSet]

				matcher.build_keypoints(
    				clear_images=True,
        			clear_keypoints=True,
        			overwrite=True,
        			masks=mask)

				matcher.build_matches(
        			maxdt=MAXDT, 
        			seq=MATCH_SEQ,
        			path=TOUNGE_MATCHES_PATH,
			        max_ratio=0.75,
			        weights=True,
			        max_distance=None
			        )

				matcher.filter_matches(clear_weights=True)
				matcher.convert_matches(glimpse.optimize.RotationMatchesXY, clear_uvs=True)

				camParams = [dict() if img.anchor else {'viewdir':([0,1,2],[-40,-45,-20],[45,45,20])} for img in matcher.images]
				cams = [image.cam for image in matcher.images]
				controls = list(matcher.matches.data)

				Cameras = glimpse.optimize.Cameras(
			    	cams = cams, 
			    	controls=controls,
			    	cam_params=camParams,
			  

			    	)

				print("\nFitting\n")
				print("Check: {} Matchers {} Controls".format(len(matcher.images),len(controls)))
				fit = Cameras.fit(method='Nelder-Mead',full=True,tol=1e-8,options={"maxiter":100,"disp":True})

		
				Cameras.set_cameras(fit.params)
				overlap = []
				[overlap.append(image) for image in subSet[-2:] ]

				subSet = []
				self.saveCams(matcher,Cameras)
				matcher = None
				Cameras = None
				fit = None
				
                              

	def run(self):
		self.modelRef()
		self.getImages()
		self.iterMatch()

	def saveCams(self,matcher,cameras):
		directory = self.savedir
		for images,newcam in  zip(matcher.images,[camera for camera in cameras.cams]):
			filename = images.path.replace("jpg","JSON")
			path = os.path.join(directory,filename)
			print("\n Writing {} \n".format(filename))
			newcam.write(path=path,attributes=("viewdir","xyz","sensorsz","c","p","k","imgsz","f"))

if __name__ == "__main__":


	
	'''
	dual2worldpts = np.array([[537464,7379214,1096],[528472.74,7370568.6,416],[537870.25,7365624.42,167],[535469.85,7364576,237]])
	dual2imgpts = np.array([[686,165],[38,228],[772,248],[515,240]])
	dual2basedict = dict(sensorsz=(5.70,4.28),xyz=(536935.8039, 7357022.081, 469.526), viewdir=(0,0,0))
	dual2savedir = "/home/dunbar/Research/helheim"
	dual2imgdir = "/home/dunbar/Research/helheim/images/helheim/stardot2"

	'''

	dual1worldpts = np.array([[537443.79,7379236.59,1086],[541582,7362810,110],[540547,7362878,119],[542619.85,7362388.83,111]])
	dual1imgpts = np.array([[59,155],[783,278],[672,261],[926,286]])
	dual1basedict = dict(sensorsz=(5.70,4.28),xyz=(536935.8039, 7357022.081, 469.526), viewdir=(0,0,0))
	dual1savedir = "/home/dunbar/Research/helheim"
	dual1imgdir = "/home/dunbar/Research/helheim/images/helheim/stardot1"
	dual1mask = '/home/dunbar/Research/helheim/dual1mask.npy'


	basemodelpath = "/home/dunbar/Research/helheim/intrinsicmodel.json"
	dual1viewopt = optView(dual1savedir,dual1imgdir,basemodelpath,dual1basedict,dual1imgpts,dual1worldpts,mask=dual1mask)
	dual1viewopt.run()

