import os
import sys

sys.path.append("/home/dunbar/Research/wolverine/wolverineglacier/scripts/glimpse")
 

import glimpse

import glimpse.svg

import glimpse.optimize

import matplotlib.pyplot as plt

import numpy as np




ROOT = "/home/dunbar/Research/helheim"

DEM_PATH = "/home/dunbar/Research/helheim/images/helheim/dem/helheim_wgs84.tiff"

IMG_DIR = os.path.join(ROOT, "stardot1", "stardot2")

#SVG_DIR = "/home/dunbar/Research/wolverine/data/cam_cliff_0496.svg"


ANCHOR = "/home/dunbar/Research/helheim/images/helheim/stardot2/HEL_DUAL_StarDot2_20200311_150000.jpg"





CAM_PARAMS = {
"viewdir": [2.842332938979837, -24.999999625888325, -7.288896327536902], 
"xyz": [536955.8038998549, 7357002.0810000235, 449.5260000000391],
 "sensorsz": [5.7, 4.28], "c": [384.0, 612.0], 
 "p": [0.0009601409206891059, -0.0009339433598518774], 
 "k": [-0.45918632488083716, 0.6732387492072175, -0.744039947588675, 0.0, 0.0, 0.0], 
 "imgsz": [768.0, 1024.0], "f": [1346.9994655875705, 1793.9999992684657]
 }








# --- Prepare anchor image ---




img = glimpse.Image(path=ANCHOR,cam=CAM_PARAMS)
img.cam.xyz = np.rint(img.cam.xyz).astype(np.int)

print(img.cam.xyz)


# --- Build points control ---
dual2worldpts = np.array([[537464,7379214,1096],[528472.74,7370568.6,416],[537870.25,7365624.42,167],[535469.85,7364576,237]])
dual2imgpts = np.array([[686,165],[38,228],[772,248],[515,240]])

points = glimpse.optimize.Points(

cam=img.cam,

uv=dual2imgpts,

xyz=dual2worldpts)



# --- Build lines (horizon) control ---




# World coordinates

dem = glimpse.Raster.read(path=DEM_PATH, d=2)
dem.crop(zlim=(0, np.inf))
dem.fill_crevasses(mask=~np.isnan(dem.Z), fill=True)

xyz = img.cam.xyz
xyzs = dem.horizon(xyz)




# Plot horizon on DEM

'''
dem.plot(cmap="Greys")

[plt.plot(xyz[:, 0], xyz[:, 1], 'r') for xyz in xyzs]

plt.plot(img.cam.xyz[0], img.cam.xyz[1],'k.')

plt.show()
'''

# Images coordinates
'''
svg = glimpse.svg.read(

path=SVG_DIR,
key="id", 
imgsz=img.cam.imgsz

)

uvs = [np.array(svg["horizon"])]




lines = glimpse.optimize.Lines(cam=img.cam, uvs=uvs, xyzs=xyzs)
'''



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
