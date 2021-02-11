import pdb
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import glimpse
import datetime,os
import numpy as np
import glob
import itertools
from os.path import join
os.environ['OMP_NUM_THREADS'] = '1'
#==============================================

DATA_DIR = "/home/dunbar/Research/helheim/data/observations"
DEM_DIR = os.path.join(DATA_DIR, 'dem')
MAX_DEPTH = 30e3

# ---- Prepare Observers ----

observerpath = ['stardot1','stardot2']
observers = []
for observer in observerpath:
    path = join(DATA_DIR,observer)
    campaths =  glob.glob(join(path,"*.JSON"))
    images = [glimpse.Image(path=campath.replace(".JSON",".jpg"),cam=glimpse.camera.Camera.from_json(campath)) for campath in campaths]
    images.sort(key= lambda img: img.datetime)
    datetimes = np.array([img.datetime for img in images])
    for n, delta in enumerate(np.diff(datetimes)):
        if delta <= datetime.timedelta(seconds=0):
            secs = datetime.timedelta(seconds= n%5 + 1)
            images[n+1].datetime = images[n+1].datetime + secs
    diffs = np.array([dt.total_seconds() for dt in np.diff(np.array([img.datetime for img in images]))])
    negate = diffs[diffs <= 1].astype(np.int)
    [images.pop(_) for _ in negate]

    images = images[:int(len(images)/2)]
    print("Image set {} \n".format(len(images)))
    obs = glimpse.Observer(list(np.array(images)),cache=False)
    observers.append(obs)
#-------------------------
testcam = observers[1].images[0].cam
xy0  = np.array([ 533528.0,7361411.0,180])[np.newaxis,:]
infront = testcam.infront(xy0)
print(f"\n Camera Location: {testcam.xyz}\n")
print(f"\n Point in Front: {infront}\n\n")
polygon = testcam.viewpoly(depth=10e4)

print(polygon.shape)
'''
path = glob.glob(join(DEM_DIR,"*.tif"))[0]
print("DEM PATH: {}".format(path))
dem = glimpse.Raster.open(path=path)

dem.crop(zlim=(0, np.inf))
dem.fill_crevasses(mask=~np.isnan(dem.array), fill=True)

# ---- Prepare viewshed ----
for obs in observers:
    dem.fill_circle(obs.images[0].cam.xyz, radius=50)
viewshed = dem.copy()
viewshed.array = np.ones(dem.shape, dtype=bool)
for obs in observers:
    viewshed.array &= dem.viewshed(obs.images[0].cam.xyz)

plt.plot(polygon[:,0],polygon[:,1])
plt.show()
'''
xy0  = np.array([ 533528.0,7361411.0,180])[np.newaxis,:]
infront = testcam.infront(xy0)
xy = xy0[:,0:2]+ np.vstack([xy for xy in itertools.product(range(-200, 200, 25), range(-200, 200, 25))])
elevs = 180*np.ones((xy.shape[0],1))

xy = np.concatenate((xy,elevs),1)

uv = testcam.xyz_to_uv(xy)
#print(uv)
observers[1].images[0].plot()
matplotlib.pyplot.scatter(uv[:, 0], uv[:, 1])
plt.show()
