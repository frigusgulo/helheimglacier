import numpy as np
import os

os.system("cp ~/Downloads/StarDot2.zip /home/dunbar/Research/helheim/images/helheim/stardot2")
os.system("cd ~/Research/helheim/images/helheim/stardot2")
os.system("unzip ~/Research/helheim/images/helheim/stardot2/StarDot2.zip")
#os.system("rm *.partial")
os.system("mv StarDot2/* .")
os.system("rm -rf StarDot2")
os.system("rm $(ls | grep 201607)")
os.system("python ~/Research/helheim/modexif.py ~/Research/helheim/images/helheim/stardot2")
os.system("rm -rf StarDot2.zip")