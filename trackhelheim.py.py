import pdb
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import glimpse
from glimpse.imports import datetime,np,os
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
    images = [glimpse.Image(path=campath.replace(".JSON",".jpg"),cam=campath) for campath in campaths]
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
dem = glimpse.Raster.read(path,d=10)
print("DEM PATH: {}".format(path))
dem.crop(zlim=(0, np.inf))
dem.fill_crevasses(mask=~np.isnan(dem.Z), fill=True)



# ---- Prepare viewshed ----

for obs in observers:
    dem.fill_circle(obs.xyz, radius=100)
viewshed = dem.copy()
viewshed.Z = np.ones(dem.shape, dtype=bool)
for obs in observers:
    viewshed.Z &= dem.viewshed(obs.xyz)

print("\n *****Viewshed Done**** \n")
# ---- Run Tracker ----
xy = []
xy0 = np.array(( 7361411.0,533528.0))
#xy.append(xy0)

xy = xy0 #qu+ np.vstack([xy for xy in itertools.product(range(-200, 200, 25), range(-200, 200, 25))])

# Helheim vels around 0.8 m/day

time_unit = datetime.timedelta(days=1)
motion_models = [glimpse.CartesianMotionModel(
    xyi, time_unit=time_unit, 
    dem=dem, 
    dem_sigma=2.5, 
    n=5000, 
    xy_sigma=(0.5,0.5),
    vxyz_sigma=(.4, .3, 0.02),
    axyz=(0,0,0), 
    axyz_sigma=(.25, .25, .05)) for xyi in xy]


# motion_models = [glimpse.CylindricalMotionModel(
#     xyi, time_unit=time_unit, dem=dem, dem_sigma=3, n=5000, xy_sigma=(2, 2),
#     vrthz_sigma=(np.sqrt(50), np.pi, 0.2), arthz_sigma=(np.sqrt(8), 0.05, 0.2))
#     for xyi in xy]
tracker = glimpse.Tracker(observers=observers, viewshed=viewshed)
print("\n****Tracking Now*****\n")
tracks = tracker.track(motion_models=motion_models, tile_size=(20, 20),
    parallel=True)
print(tracks.errors)

# ---- Plot tracks ----
#tracks.plot_v1d(dim=1,mean=True)
tracks.plot_xy(start=dict(color='green'), mean=dict(color='red'), sigma=dict(alpha=0.25))
plt.show()
