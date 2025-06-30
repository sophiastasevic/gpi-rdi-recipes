#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 10:48:56 2023

@author: sstas

1) reads in data paths from the sof file + matches data type

2) identifies path belonging to the science cubes

3) compiles header and star information of cubes

4) removes paths from the reference cube list belonging to:
  a) missing files
  b) observations with waffle on
  c) duplicate reductions of the same observation (keeps most recent reduction with most frames)
  d) other epoch observations of the science target
  e) targets with known companion/visible binary in FOV/debris disk or non stellar targets
  
5) matches data cubes to their respective frame selection vector

6) matches timestamp and frame rotation angle files to each cube
   
7) [optional] removes the science cube from the reference cube list
    
8) creates fits table containing reference and science cube inforation
    
"""
import warnings
import os, re
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.utils.exceptions import AstropyWarning
from target_info_table import star_info, init_info_dict, add_target_info, make_ref_table
from retry_handler import RetryHandler

warnings.simplefilter('ignore', category=AstropyWarning)

cut_path = lambda path: path.split('/dwh/')[-1]
"""
---------------------------------- FUNCTIONS ----------------------------------
"""
def read_file(func, path, **kwargs):
    if "summer" in path:
        return RetryHandler(func).run(path, **kwargs)
    else:
        return func(path, **kwargs)


## match the cube path with the same path directory as the lambda path
def match_lambda(cube_paths, lambda_path):
    sci_dir = os.path.dirname(lambda_path)
    cube_dirs = [os.path.dirname(x) for x in cube_paths]
    try:
        ## get the index of the lambda path dir in the list of cube dirs
        ## then select the cube path at that same index
        science_path = cube_paths[cube_dirs.index(sci_dir)]
    except IndexError:
        ## no cube path with a matching dir, stop process
        raise Exception('> No GPI_REDUCED_COLLAPSED_MASTER_CUBE in recipe input matching GPI_SCIENCE_LAMBDA_INFO path directory: {0:s}'.format(sci_dir))
        
    return science_path


## reading header information from input frame selection vectors to match to
## data cubes + removes paths with missing files
def match_frame_select(frame_paths_tmp, target_epochs, nframes):
    frame_id_tmp = []
    frame_date = []
    frame_nframe = []
    problem_frames = []
    for frame_path in frame_paths_tmp:
        try:
            frame_hdr = read_file(fits.getheader, frame_path)
            frame_id_tmp.append(frame_hdr['ESO OBS START'])
            frame_date.append(frame_hdr['DATE'])
            frame_nframe.append(frame_hdr['NAXIS{0:d}'.format(frame_hdr['NAXIS'])])
        except FileNotFoundError:
            print('> Warning: Could not find frame selection vector file:', frame_path)
            problem_frames.append(frame_path)
    frame_paths_tmp = [x for x in frame_paths_tmp if x not in problem_frames]
    
    ## matches each data cube to its frame selection vector
    frame_id = []
    frame_paths = []
    for ei, epoch in enumerate(target_epochs):
        frame_id.append(epoch)
        framei = [i for i,x in enumerate(frame_id_tmp) if x==epoch and frame_nframe[i]==nframes[ei]]
        if len(framei)==0:
            frame_paths.append('na')
            continue
        if len(framei)==1:
            framei=framei[0]
        elif len(framei)>1:
            process_date = [x for i,x in enumerate(frame_date) if i in framei]
            try:
                framei = framei[process_date.index(frame_date[ei])]
            except ValueError:
                framei = framei[process_date.index(max(process_date))]
        frame_paths.append(frame_paths_tmp[framei])
           
    return frame_paths


## removing duplicate reducions of the same observaion
def remove_dupes(target_id, science_id, isci, nframes, target_dates, cube_paths):
    elements = {}
    for target in target_id: 
        if elements.get(target,None) != None:
            elements[target]+= 1
        else:
            elements[target] = 1
    
    dupes = [k for k,v in elements.items() if v>1]   
    problem_stars = []
    for target in dupes:
        index = [i for i,x in enumerate(target_id) if x==target]
        if target == science_id:
            keep = isci
            print('> Removing duplicate reductions of science target:', target) 
        else:
            frame_count = nframes[index] ## checks first if there are duplicate reductions that have fewer frames than the others
            rm = [i for i,x in zip(index,frame_count) if x!=frame_count.max()]
            process_date = [x for i,x in enumerate(target_dates) if i in index and i not in rm]
            keep = index[process_date.index(max(process_date))] ## most recently reduced
            print('> Warning: Multiple reductions of same observation found for:', 
                  target, 'Keeping most recent with', max(frame_count), 'frames.')    
        problem_stars += [cube_paths[i] for i in index if i!=keep]
        
    return problem_stars


## remove reference paths belonging to other observations of the science target
## and the science path if not being used as a reference
def science_dupes(isci, science_path, target_names, target_epochs, cube_paths, USE_SCI):
    problem_stars = []
    if USE_SCI == False: 
        problem_stars.append(science_path)
            
    for i,target in enumerate(target_names):
        if target == target_names[isci] and target_epochs[i] != target_epochs[isci]:
            problem_stars.append(cube_paths[i])
            print('> Removing {0:s} observation of science target: {1:s} {2:s}' \
                  .format(target_epochs[i], target_names[isci], target_epochs[isci]))
            
    return problem_stars


def remove_known_signal(target_names, data, cube_paths, keep=[]):
    known_signal = data.loc[(data['binary']==True)|
                            (data['planet']==True)|
                            (data['disk']==True)].index
    return [cube_paths[i] for i,name in enumerate(target_names) if name in known_signal and cube_paths[i] not in keep]
     

def get_constants_paths(fnames, ftypes):
    columns_path = fnames[np.where(ftypes=="GPI_RDI_TABLE_COLUMN_META")[0]][0]
    signal_paths = fnames[np.where(ftypes=="GPI_RDI_REFERENCE_TARGET_FLAG")[0]]
    
    return columns_path, signal_paths

def get_sof_paths(fnames, ftypes, frame, nmin):
    path = fnames[np.where(ftypes==frame)[0]]
    if len(path) < nmin:
        raise Exception('[Error] Input must contain at least', nmin, frame)
    
    return path

def read_sof(sof, data_names):
    data = np.loadtxt(sof, dtype=str)
    fnames = data[:,0] #file names
    ftypes = data[:,1] #file data types

    paths = {}
    for var in data_names.keys():
        paths[var] = get_sof_paths(fnames, ftypes, *data_names[var])
    columns_path, signal_paths = get_constants_paths(fnames, ftypes)
    
    return paths, columns_path, signal_paths


## load list of objects with a known companion, disk, or visible binary
def get_companion_disk_flag(signal_paths):
    paths = {}
    for k in ["disk", "planet", "binary", "non_stellar"]:
        for p in signal_paths:
            if k in p: paths[k] = p
    known_signal = {}
    for col in paths.keys():
        data = np.loadtxt(paths[col], dtype=str, delimiter='\t')
        if data.ndim > 1:
            known_signal[col] = [str(x) for x in data[:,0]]
        else:
            known_signal[col] = [str(x) for x in data]
    return known_signal


## match science cube path using lambda path and create dict of science paths + header information
## to be used for the ref_table header later
def get_science_paths(data_paths):
    lambda_path = data_paths['lambda'][0]
    sci_paths = {'cube': match_lambda(data_paths['cube'], lambda_path), 'lambda': lambda_path}
    sci_paths['isci_data_paths'] = list(data_paths['cube']).index(sci_paths['cube'])
    
    ## if the science cube is in target_cache, it's header won't be read in later, so read in now
    sci_hdr = read_file(fits.getheader, sci_paths['cube'])
    header_dict = {k:sci_hdr[k] for k in ['ESO INS COMB ICOR', 'ESO INS COMB IFLT']}
    header_dict['SCI_ID'] = sci_hdr['DATASUM']
    header_dict['SCI_PATH'] = sci_paths['cube']
    
    return sci_paths, header_dict
         
## reading in + sorting file paths read from sof file
def get_paths(sofname, data_names, USE_SCI):
    data_paths, col_path, signal_paths = read_sof(sofname, data_names)
    ncubes = len(data_paths['cube'])
    
    sci_paths, header_dict = get_science_paths(data_paths)
    
    ## list of objects observed by SPHERE that are not stars
    known_signal = get_companion_disk_flag(signal_paths)
    non_stars = known_signal.pop("non_stellar")
    
    ## previous target info table, if given as an input
    try:
        target_cache = read_file(Table.read, data_paths['target_table'][0])
        target_cache['Path'] = [cut_path(x) for x in target_cache['Path']]
        target_cache.add_index('Path')
        print("> Taking target data from GPI_REFERENCE_TABLE output by a previous run of this recipe.")
        
    except(KeyError, IndexError):
        print("> Querying star-server for each GPI_REDUCED_COLLAPSED_MASTER_CUBE input.")
        target_cache = None
        
    """
    csv file containing: 
        table column name [index]
        header name (if appl.)
        sparta query name (if appl.)
        data type
        unit
        data source (i.e., header, sparta query, etc.)  
    """
    columns = pd.read_csv(col_path, index_col=0) 
    ## dictionary that stores header information of each cube
    target_info = init_info_dict(ncubes, columns, list(data_paths['cube']))
    
    problem_stars = []
    star_data = None
    
    ## reading header information from input cubes for later sorting + removes
    ## paths whose files cannot be found + finds path of science cube
    for i,path in enumerate(data_paths['cube']):
        path_tmp = cut_path(path)
        if target_cache is not None and path_tmp in list(target_cache['Path']):
            ## previous target info table loaded in recipe and path exists in table
            cache = target_cache.loc[path_tmp]
            if cache['Simbad_ID'] == '': ## either waffle on or file not found
                problem_stars.append(path)
            else:
                sid, star_data = star_info(star_data, known_signal, columns, cache=cache)
                add_target_info(target_info, columns, i, sid, cache=cache)
            continue
        
        try:
            hdr_tmp = read_file(fits.getheader, path)
            if hdr_tmp['OBJECT'] in non_stars:
                problem_stars.append(path)
                continue
            
            try:
                waffle_on = 'CENTER' in hdr_tmp['ESO DPR TYPE']
            except KeyError:
                waffle_on = False
            if hdr_tmp['DATE-OBS'] == '2019-03-23T09:20:29.73': 
                waffle_on = True
                
            if waffle_on:
                if path == sci_paths['cube']:
                    raise Exception('> Science target observed with waffle on')
                else:
                    problem_stars.append(path)
                    continue
            
            if target_cache is not None:
                print("> Querying star-server for new GPI_REDUCED_COLLAPSED_MASTER_CUBE input:", hdr_tmp['OBJECT'], hdr_tmp['DATE-OBS'])
            ## ESO OBS START is a more reliable identifier than DATE-OBS  as it is the same for any
            ## convert process of the same observation (whereas DATE-OBS can vary by a few minutes)
            sid, star_data = star_info(star_data, known_signal, columns, hdr_tmp['RA'], hdr_tmp['Dec'], 
                                       hdr_tmp['OBJECT'], hdr_tmp['ESO OBS START'])
            
            add_target_info(target_info, columns, i, sid, hdr_tmp)
            
        except FileNotFoundError:
            if path == sci_paths['cube']:
                raise Exception('> Unable to find science target GPI_REDUCED_COLLAPSED_MASTER_CUBE file: {0:s}.',format(path))
            print('> Warning: Could not find data cube file:', path)
            problem_stars.append(path)
    
    star_data = star_data.set_index('simbad main identifier')
    ## all cubes that exist and don't have waffle
    cube_paths = [x for x in data_paths['cube'] if x not in problem_stars]
    
    ## regex search for text in the format xxxx-xx-xx where x is any digit between 0 and 9 
    p_date = re.compile(r'[0-9]{4}-[0-9]{2}-[0-9]{2}')
    ## dates that the cubes were created in the DC
    cube_dates = p_date.findall(' '.join(cube_paths))
    cube_data = {col:[target_info[col][target_info['Path'].index(x)] for x in cube_paths]
                 for col in ['ESO_obs_start','Simbad_ID','Cube_Nframes']}
    cube_data['Cube_Nframes'] = np.array(cube_data['Cube_Nframes'], dtype=int)
    
    frame_paths = match_frame_select(data_paths['frame'], cube_data['ESO_obs_start'], cube_data['Cube_Nframes'])
           
    isci = cube_paths.index(sci_paths['cube'])
    sci_epoch = cube_data['ESO_obs_start'][isci]
    
    ## remove duplicate reductions of the same observation
    problem_stars += remove_dupes(cube_data['ESO_obs_start'], sci_epoch, isci, cube_data['Cube_Nframes'], cube_dates, cube_paths)
    
    ## remove other observations of the science target + science target if USE_SCI==False
    problem_stars += science_dupes(isci, sci_paths['cube'], cube_data['Simbad_ID'], cube_data['ESO_obs_start'], cube_paths, USE_SCI)   
    
    nrm=len(problem_stars)
    ## remove ref targets with known disks/companions/visible binaries (except science target if USE_SCI==True)
    problem_stars += remove_known_signal(cube_data['Simbad_ID'], star_data, cube_paths, keep=[sci_paths['cube']])
    print(len(problem_stars)-nrm, "reference targets removed for having known disks/companions")
    
    ## complete dict of cube paths with corresponding frame selection vector, rotation vector, and timestamp paths
    ref_paths = {'cube': [x for x in cube_paths if x not in problem_stars],
                 'frame': [x for i,x in enumerate(frame_paths) if cube_paths[i] not in problem_stars],
                 'frame_cache': data_paths['frame_data']}
    
    sci_paths['frame'] = frame_paths[isci]
    
    ref_flag = [False if x in problem_stars else True for x in data_paths['cube']]
    ## convert dictionary to fits table of reference targets with their star information
    
    target_table = make_ref_table(target_info, columns, star_data, header_dict, ref_flag,
                                  sci_paths.pop('isci_data_paths'), USE_SCI)
    
    return sci_paths, ref_paths, target_table