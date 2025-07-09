#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 21:18:12 2025

@author: sstas
"""

import os
import warnings
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning

from retry_handler import RetryHandler

warnings.simplefilter('ignore', category=AstropyWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

"""
---------------------------------- FUNCTIONS ----------------------------------
"""
def read_file(func, path, **kwargs):
    """
    Handle reading in files if data is taken from summer.
    """
    if "summer" in path:
        return RetryHandler(func).run(path, **kwargs)
    else:
        return func(path, **kwargs)

def get_sof_paths(fnames, ftypes, frame, nmin):
    """
    Return paths in sof belonging to the requested frame type
    """
    path = fnames[np.where(ftypes==frame)[0]]
    if len(path) < nmin:
        raise Exception('[Error] Input must contain at least', nmin, frame)
        
    return path

def read_sof(sof, data_names):
    """
    Read sof file and return list of paths for each frame type
    """
    data = np.loadtxt(sof, dtype=str)
    fnames = data[:,0] #file names
    ftypes = data[:,1] #file data types

    paths = {}
    for var in data_names.keys():
        paths[var] = get_sof_paths(fnames, ftypes, *data_names[var])
    
    return paths

def format_df(paths, science_epoch):
    """
    Read in target info CSV and check that cube order and number of cubes matches the correlation matrix.

    Parameters
    ----------
    paths : dict
        Input data paths
    science_epoch : str
        Observation date of science cube

    Returns
    -------
    target_data : DataFrame
        Stellar parameters, header information, and paths of all GPI cubes
    isci : int
        Index position science cube in full cube list
    """
    
    corr_hdr = read_file(fits.getheader, paths['corr'])
    ncubes = corr_hdr['NAXIS1']
    epochs = [corr_hdr["OBS_{0:04d}".format(i)] for i in range(ncubes)]
    
    ## get position of science cube in list
    try:
        isci = epochs.index(science_epoch)
    except ValueError:
        raise Exception('[Error] Science cube not in correlation matrix.')
    
    ## data frame containing paths of each gpi cube and frame selection vector
    target_data = read_file(pd.read_csv, paths['corr_data'], index_col='date-obs')
    
    ## check that the order of the cubes in the header is the same as in the df
    if target_data.shape[0] != ncubes:
        raise Exception('[Error] Number of GPI_REFERENCE_TARGET_DATA cubes (%d) does not match GPI_CORR_MATRIX (%d).' % (target_data.shape[0], ncubes))
    
    elif list(target_data.index) != epochs:
        target_data.reindex(index=epochs, inplace=True)
    
    for col in ['path', 'frame_path']:
        target_data[col] = check_paths(list(target_data[col]), paths['science'])
    
    ## True/False flag for where path == 'na' (i.e. file does not exist)
    target_data['bad_path'] = target_data['path'].eq('na')
    
    return target_data.reset_index(), isci


def check_paths(paths_tmp, science_path):
    """
    Check if paths exist and format directory if using summer
    """
    if "summer" not in science_path and "summer" in paths_tmp[0]:
        paths_tmp = ["/dwh/" + p.split('/dwh/')[-1] for p in paths_tmp]
    
    elif "summer" in science_path and "summer" not in paths_tmp[0]:
        paths_tmp = ["/summer/sphere/data_ext" + p for p in paths_tmp]
    
    data_ext = ["data%d_ext" % i for i in range(1,6)] + ["data_ext_cobrex%s" % x for x in ["", "_2"]]
    
    for i, p in enumerate(paths_tmp):
        if p != "na" and read_file(os.path.exists, p) == False: 
            if "data_ext" in p:
                ## try different data extensions
                for ext in data_ext:
                    p_try = p.replace("data_ext", ext)
                    
                    if read_file(os.path.exists, p_try) == True:
                        paths_tmp[i] = p_try
                        break
                    else:
                        paths_tmp[i] = "na"
                        print('> Warning: Could not find file:', p)
            else:
                paths_tmp[i] = "na"
                print('> Warning: Could not find file:', p)
    
    return paths_tmp