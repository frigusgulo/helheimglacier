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
uv = observer[1].images[0].cam.project(( 7361411.0,533528.0,180))
observer[1].images[0].cam.plot()
matplotlib.pyplot.scatter(uv[:, 0], uv[:, 1])
