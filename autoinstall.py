import os, time

os.system('conda env create -f conda-env-conf.yml')
time.sleep(5000)
os.system('conda activate bspyalgo')
time.sleep(10000)
os.system('python setup.py develop')
