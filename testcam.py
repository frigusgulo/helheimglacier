import os
import sys

sys.path.append("/home/dunbar/Research/wolverine/wolverineglacier/scripts/glimpse")
 

import glimpse

import glimpse.svg

import glimpse.optimize

import matplotlib.pyplot as plt

import numpy as np




#ROOT = "/home/dunbar/Research/wolverine/data/"

DEM_PATH = "/home/dunbar/Research/helheim/data/observations/dem/helheim_wgs84UTM.tif"

#IMG_DIR = os.path.join(ROOT, "cam_cliff", "images")

#SVG_DIR = "/home/dunbar/Research/wolverine/data/cam_cliff_0496.svg"


ANCHOR = "/home/dunbar/Research/helheim/data/observations/stardot2/HEL_DUAL_StarDot2_20200311_150000.jpg"





CAM_PARAMS = {

"viewdir": [-2.3512759020041543, -30.189334073000776, 4.256111678804434], 

"xyz": [536955.8030958388, 7357002.081213994, 450.1814980033436], 

"sensorsz": [5.7, 4.28], 

"c": [100.0, 0.0],

 "p": [0.0009601409206891059, -0.0009339433598518774], 

 "k": [-0.45918632488083716, 0.6732387492072175, -0.744039947588675, 0.0, 0.0, 0.0], 

 "imgsz": [1024.0, 768.0], 

 "f": [1793.9000490540586, 1338.3455527807855]
}







# --- Prepare anchor image ---




img = glimpse.Image(path=ANCHOR,cam=CAM_PARAMS)




# --- Build points control ---
'''
#cliffimgpts = np.array([[5193, 549],[3101, 642]])#,[6153.0, 2297.0]])
#cliffworldpts = np.array([[408245.86,6695847.03,1560 ],[416067.22,6707259.97,988]])#,[394569.509, 6695550.678, 621.075]])



points = glimpse.optimize.Points(

cam=img.cam,

uv=cliffimgpts,

xyz=cliffworldpts)



# --- Build lines (horizon) control ---


'''

# World coordinates

dem = glimpse.Raster.read(path=DEM_PATH, d=10)

print(img.cam.xyz)
xyzs = dem.horizon(img.cam.xyz)




# Plot horizon on DEM

dem.plot(cmap="Greys")

[plt.plot(xyz[:, 0], xyz[:, 1], 'r') for xyz in xyzs]

plt.plot(img.cam.xyz[0], img.cam.xyz[1],'k.')

plt.show()
'''

# Images coordinates
svg = glimpse.svg.read(

path=SVG_DIR,
key="id", 
imgsz=img.cam.imgsz

)

uvs = [np.array(svg["horizon"])]




lines = glimpse.optimize.Lines(cam=img.cam, uvs=uvs, xyzs=xyzs)




# Plot horizon on image

img.plot()

lines.plot()

img.set_plot_limits()

plt.show()

# --- Calibrate camera ---



model = glimpse.optimize.Cameras(

cams=[img.cam],

controls=[lines,points], #removed "points" control

cam_params=[{"viewdir": True,"f":True}],
)

params = model.fit()

print("\n\n Params: \n {}".format(params))



# Plot controls on image (uncalibrated)

img.plot()

model.plot()

img.set_plot_limits()

plt.show()



# Plot controls on image (calibrated)

img.plot()

model.plot(params)

img.set_plot_limits()

plt.show()
'''