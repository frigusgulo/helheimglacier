import numpy as np
import glob
import sys
import json
from os.path import join

directory = sys.argv[1]

files = glob.glob(join(directory,"*.JSON"))
for file in files:
	with open(file) as f:
		model = json.load(f)
	
	c = model["c"]
	imgsz = model["imgsz"]

	c[0] = np.abs(np.rint(imgsz[0]//2) - 483)
	c[1] = np.abs(np.rint(imgsz[1]//2)-413)	
	f = (1075.4,1095.8)
	model["xyz"][-1] = 430
	model["imgsz"] = imgsz
	model["f"] = f
	model["k"] = [-4.32e-01,  4.57e-01,0,  0, -6.88192e-01]
	p = [ -2.5e-03,  -4.021e-05]
	model["p"] = p
	with open(file,'w') as f:
		json.dump(model,f)
