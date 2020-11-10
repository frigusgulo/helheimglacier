from datetime import datetime
import piexif
import glob
import sys
import os
import numpy as np


jpgFolder = sys.argv[1]
## count number of photos found
listOfFiles = glob.glob(os.path.join(jpgFolder, "*.jpg"))
## create datetimeString from JPG filename
for jpg in listOfFiles:
    date = jpg.split("_")[3]
    time = jpg.split("_")[4]

    year = np.int(date[0:4])
    month = np.int(date[4:6])
    day = np.int(date[6:])
    hour = np.int(time[0:2])
    mins = np.int(time[2:4])
    #datetimeStringNew = day + "/" + month + "/" + year + " " + hour + ":" + mins

## change exif datetimestamp for "Date Taken"
    exif_dict = {}
    exif_dict['Exif'] = { piexif.ExifIFD.DateTimeOriginal: datetime(year, month, day, hour, mins,00).strftime("%Y:%m:%d %H:%M:%S") }
    exif_bytes = piexif.dump(exif_dict)
    piexif.insert(exif_bytes, jpg)