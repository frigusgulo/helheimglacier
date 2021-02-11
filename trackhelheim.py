import pdb
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import glimpse
import numpy as np
import datetime
import os

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
    images = [glimpse.Image(path=campath.replace(".JSON",".jpg"),cam=glimpse.Camera.from_json(campath)) for campath in campaths]
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


#----------------------------
# Prepare DEM 
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

print("\n *****Viewshed Done**** \n")
# ---- Run Tracker ----
xy = []
xy0 = np.array([ 537000,7361500])[np.newaxis,:]
#xy.append(xy0)

xy = [xy0] #qu+ np.vstack([xy for xy in itertools.product(range(-200, 200, 25), range(-200, 200, 25))])

# Helheim vels around 0.8 m/day

time_unit = datetime.timedelta(days=1)
motion_models = [glimpse.CartesianMotion(
    xyi, time_unit=time_unit, 
    dem=dem, 
    dem_sigma=2.5, 
    n=5000, 
    xy_sigma=(0.5,0.5),
    vxyz_sigma=(.4, .3, 0.02),
    axyz=(0,0,0), 
    axyz_sigma=(.25, .25, .05)) for xyi in xy]

tracker = glimpse.Tracker(observers=observers, viewshed=viewshed)
print("\n****Tracking Now*****\n")
tracks = tracker.track(motion_models=motion_models, parallel=True)
print(tracks.errors)

# ---- Plot tracks ----
#tracks.plot_v1d(dim=1,mean=True)
tracks.plot_xy(start=dict(color='green'), mean=dict(color='red'), sigma=dict(alpha=0.25))
plt.show()
