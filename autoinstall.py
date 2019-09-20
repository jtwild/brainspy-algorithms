import os

os.system('conda env create -f conda-env-conf.yml')
os.system('conda activate bspyalgo')
os.system('python setup.py develop')
