import os, time
import subprocess as sp


print('Starting autoinstall')
print('Creating conda environment')
cep = sp.Popen(['conda env create -f conda-env-conf.yml'], bufsize=2048, shell=True,
    stdin=sp.PIPE, stdout=sp.PIPE, close_fds=True)
cep.wait()
print('Activating conda environment')
os.system('conda activate bspyalgo')
time.sleep(200)
os.system('python setup.py develop')
