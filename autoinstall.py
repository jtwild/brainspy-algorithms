import os, time
import subprocess as sp

print('Activating the environment')
os.system('conda activate bspysmg')
time.sleep(2)
print('General setup and registering the folder into the environment path')
os.system('python setup.py develop')
