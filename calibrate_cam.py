import numpy as np
import cv2 as cv
import glob
import glimpse
import  glimpse.convert as gc

import matplotlib.pyplot as plt
import sys
import os
import matplotlib.pyplot as plt
def parse_camera_matrix(x):
    """
    Return fx, fy, cx, and cy from camera matrix.
    Arguments:
        x (array-like): Camera matrix [[fx 0 cx], [0 fy cy], [0 0 1]]
    
    Returns:
        dict: fx, fy, cx, and cy
    """
    x = np.asarray(x)
    return {'fx': x[0, 0], 'fy': x[1, 1], 'cx': x[0, 2], 'cy': x[1, 2]}


def parse_distortion_coefficients(x):

    x = np.asarray(x)
    labels = ('k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6')
    return {key: x[i] if i < len(x) else 0 for i, key in enumerate(labels)}

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
chessboard_flags = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
dimx = 9
dimy = 6
edgesizeMeters = .0204216 # milimeters
objp = np.zeros((dimx*dimy,3), np.float32)
objp[:,:2] = np.mgrid[0:dimx,0:dimy].T.reshape(-1,2)*edgesizeMeters
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


dir_ = sys.argv[1]
images = glob.glob(os.path.join(dir_,'*.jpeg'))
print("Found ",len(images)," Images")
print("Dims: {} {}".format(dimx,dimy))
count = 0
for i, fname in enumerate(images):
   # print(fname,"\n")
    img = cv.imread(fname)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (dimx,dimy),None)
    # If found, add object points, image points (after refining them)
    #print("Corner status: ",ret)
    keep = False
    if ret == True:
        count += 1

        result_name = 'board'+"_"+fname
        cv.imwrite(result_name, gray)
        objpoints.append(objp)
        corners = np.squeeze(corners)
        corners = cv.cornerSubPix(gray,corners, (3,3), (-1,-1), criteria)
        imgpoints.append(corners)
        '''
        if i == (len(images)-1):
            img = cv.drawChessboardCorners(gray, (dimx, dimy), corners, ret)
            cv.imshow('img',img)
            cv.waitKey(300)
            cv.imwrite("chessboard.jpg",img)
    '''

print( "{}/{} Good Images Used".format(count,len(images)) )
        
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("Camera Matrix: ",mtx)
print("Distortion Coefficients: ",dist)
'''
cam_path = "/home/dunbar/Research/helheim/data/observations/stardot2/HEL_DUAL_StarDot2_20200311_150000.jpg"

cam = glimpse.Image(cam_path,exif=glimpse.Exif(cam_path),cam=dict(sensorsz=(5.7, 4.28))).cam
cam_matdict = parse_camera_matrix(mtx)
cam_distcoefs = parse_distortion_coefficients(np.squeeze(dist))
cammodel = cam_matdict.update(cam_distcoefs)
#cammodel = cam.to_dict().update(cammodel)

#cammodel["sensorsz"] = (5.7, 4.28)
cammodel["f"][0] = cammodel["fx"]
cammodel["f"][1] = cammodel["fy"]
cammodel.pop("fx")
cammodel.pop("fy")
cammodel["c"][0] = cammodel["cx"] 
cammodel["c"][1] = cammodel["cy"] 
cammodel["p"][0] = cammodel["p1"]
cammodel["p"][1] = cammodel["p2"]
cammodel["k"][0] = cammodel["k1"]
cammodel["k"][1] = cammodel["k2"] 
cammodel["k"][2] = cammodel["k3"]
cammodel.pop("cy")
cammodel.pop("cx")
cammodel.pop("p1")
cammodel.pop("p2")
cammodel.pop("k1")
cammodel.pop("k2")
cammodel.pop("k3")
cammodel.pop("k4")
cammodel.pop("k5")
cammodel.pop("k6")
camera = glimpse.Camera(**cammodel)
camera.write("intrinsicmodel.json")

'''