'''
Library that handles saving all relevant data.
'''
import numpy as np
import math
import time
import os
from shutil import copyfile
import sys


def save_arrays(filepath, **kwargs):
    '''
    Saves an arbitrary amount of numpy arrays in a numpy array archive named
    data. Saves it at the location specified by filepath.
    '''
    if 'filename' in kwargs:
        filename = kwargs['filename']
    else:
        filename = 'data'
    np.savez(os.path.join(filepath, filename), **kwargs)


def copy_files(filepath):
    '''
    This function copies any .py files in the script directory to the directory
    specified by filepath.
    '''

    scriptdir_stripped = sys.argv[0].split('/')[:-1]
    scriptdir = os.sep.join(scriptdir_stripped)
    if(scriptdir == ''):
        filenames = os.listdir()
    else:
        filenames = os.listdir(scriptdir)
    for filename in filenames:
        full_path = os.path.join(scriptdir, filename)
        full_path = full_path.encode('unicode_escape')
        if(os.path.isfile(full_path)):
            copyfile(full_path, os.path.join(filepath, filename))


def save_experiment(filepath, **kwargs):
    '''
    This function saves all .py files and any numpy arrays specified by **kwargs
    in the directory specified by filepath.
    '''
    save_arrays(filepath, **kwargs)
    copy_files(filepath)


def create_save_directory(filepath, name):
    '''
    This function checks if there exists a directory filepath+datetime_name.
    If not it will create it and return this path.
    '''
    datetime = time.strftime("%Y_%m_%d_%H%M%S")

    filepath = filepath + datetime + '_' + name

    if not os.path.exists(filepath):
        os.makedirs(filepath)
    return filepath
