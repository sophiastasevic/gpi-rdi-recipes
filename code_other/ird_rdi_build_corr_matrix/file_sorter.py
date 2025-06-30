#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 10:48:56 2023

@author: sstas

1) reads in data paths from the sof file + matches data type
2) removes paths belonging to:
  a) missing files
  b) targets with known companion/visible binary in FOV/debris disk or non stellar targets
  c) observations with waffle on
  d) duplicate reductions of the same observation (keeps most recent with max frames)
  
3) matches data cubes to their respective frame selection vector

4) identifies paths belonging to the science target(s)

5) removes (if 1 science target) or flags (if multiple science targets)
   paths belonging to science target observations at different epochs
   (path of the target epoch will not be removed is 'USE_SCI' == True)
   
6) matches timestamp and frame rotation angle files to each cube

"""

import warnings
import os
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
from rdi_ref_table import star_info
from retry_handler import RetryHandler

bad_centering = ['2017-06-21T04:44:22.88',
                 '2017-06-23T06:31:55.75',
                 '2017-06-23T06:53:45.73',
                 '2017-06-24T10:08:00.55',
                 '2017-06-24T10:08:01.22',
                 '2017-07-23T03:55:17.22',
                 '2018-10-12T08:52:21.44',
                 '2019-06-14T07:13:31.32',
                 '2019-10-02T04:02:31.62',
                 '2020-11-15T08:13:52.63',
                 '2021-09-21T00:59:29.44',
                 '2021-09-21T01:06:10.10']

warnings.simplefilter('ignore', category=AstropyWarning)
"""
---------------------------------- FUNCTIONS ----------------------------------
"""
def read_file(func, path, **kwargs):
    if "summer" in path:
        return RetryHandler(func).run(path, **kwargs)
    else:
        return func(path, **kwargs)

##match timestamp or frame rotation vector file to cube paths
##(assumes all cube paths have a matching timestamp or parang path in the inputs)
def match_convert(cube_paths, paths):
    if 'summer' in cube_paths[0]:
        cube_paths_tmp = [x.split('/dwh/')[1] for x in cube_paths]
        paths_tmp = [x.split('/dwh/')[1] for x in paths]
    else:
        cube_paths_tmp = cube_paths
        paths_tmp = paths
    
    cube_dirs = [os.path.dirname(x) for x in cube_paths_tmp]
    match_dirs = [os.path.dirname(x) for x in paths_tmp]
    
    index = [match_dirs.index(d) for d in cube_dirs]
    match_paths = [paths[i] for i in index]
    
    return match_paths


##for lambda files that didn't have a corresponding science path in the first check
def match_lambda(cube_paths, lambda_paths, target_epochs, target_dates, nframes):
    cube_dirs = [os.path.dirname(x) for x in cube_paths]
    unmatched_lambda_paths = [x for x in lambda_paths if os.path.dirname(x) not in cube_dirs]
    science_paths = []
    for path in unmatched_lambda_paths:
        try:
            hdr_tmp = read_file(fits.getheader, path)
            lambda_id = [hdr_tmp['OBJECT'],hdr_tmp['ESO OBS START']]
            matching_cube = [i for i,x in enumerate(target_epochs) if x==lambda_id[1]]
            
            if len(matching_cube) == 0:
                raise Exception('> IRD_SCIENCE_REDUCED_MASTER_CUBE for science target: {0:s} not in recipe input.'.format(' '.join(lambda_id)))
            elif len(matching_cube) == 1:
                cubei = matching_cube[0]
            else:
                match_nframes = [i for i in matching_cube if nframes[i]==max(nframes[matching_cube])]
                match_date = [target_dates[i] for i in match_nframes]
                cubei = match_nframes[match_date.index(max(match_date))]
            science_paths.append(cube_paths[cubei])
                
        except FileNotFoundError:
            raise Exception('> Unable to find IRD_SCIENCE_REDUCED_LAMBDA_INFO: {0:s}.'.format(path))        
    
    return science_paths


##reading header information from input frame selection vectors to match to
##data cubes + removes paths with missing files
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
    
    ##matches each data cube to its frame selection vector
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


##removing duplicate reducions of the same observaion
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
        if target in science_id:
            keep = isci[science_id.index(target)]
            print('> Removing duplicate reductions of science target:', target) 
        else:
            frame_count = nframes[index] ##checks first if there are duplicate reductions that have fewer frames than the others
            rm = [i for i,x in zip(index,frame_count) if x!=frame_count.max()]
            process_date = [x for i,x in enumerate(target_dates) if i in index and i not in rm]
            keep = index[process_date.index(max(process_date))] ##most recently reduced
            print('> Warning: Multiple reductions of same observation found for:', 
                  target, 'Keeping most recent with', max(frame_count), 'frames.')    
        problem_stars += [cube_paths[i] for i in index if i!=keep]
        
    return problem_stars


##create dict of paths belonging to observations of each science target so they can be skipped
def science_dupes(isci, science_paths, target_names, target_epochs, cube_paths, has_signal, USE_SCI):
    skip_sci_target = {}
    problem_stars = []
    for s in range(len(isci)):
        skip_sci_target[s]=[]
        known_signal_sci = [path for path in has_signal if path != science_paths[s]]
        if USE_SCI == False: 
            if len(isci)==1: 
                problem_stars.append(science_paths[s])
            else: 
                skip_sci_target[s].append(science_paths[s])
        ## skip other science targets with known signals while keeping them in the ref
        ## cube list if self correlation is being calculated
        elif USE_SCI == True and len(known_signal_sci) > 0:
            skip_sci_target[s] += known_signal_sci
            
        for i,target in enumerate(target_names):
            if target == target_names[isci[s]] and target_epochs[i] != target_epochs[isci[s]]:
                if len(isci)==1:
                    problem_stars.append(cube_paths[i])
                else:
                    skip_sci_target[s].append(cube_paths[i])
                print('> Skipping {0:s} observation for science target: {1:s} {2:s}' \
                      .format(target_epochs[i], target_names[isci[s]], target_epochs[isci[s]]))
                
    return problem_stars, skip_sci_target


def remove_known_signal(target_names, data, cube_paths, keep=[]):
    known_signal = data.loc[(data['binary']==True)|
                            (data['planet']==True)|
                            (data['disk']==True)].index
    return [cube_paths[i] for i,name in enumerate(target_names) if name in known_signal and cube_paths[i] not in keep]
     

def get_constants_paths(fnames, ftypes):
    columns_path = fnames[np.where(ftypes=="IRD_RDI_TABLE_COLUMN_META")[0]][0]
    signal_paths = fnames[np.where(ftypes=="IRD_RDI_REFERENCE_TARGET_FLAG")[0]]
    
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
    
        
## reading in + sorting file paths read from sof file
def get_paths(sofname, data_names, USE_SCI):
    data_paths, col_path, signal_paths = read_sof(sofname, data_names)
    ## list of objects observed by SPHERE that are not stars
    known_signal = get_companion_disk_flag(signal_paths)
    non_stars = known_signal.pop("non_stellar")
    
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
    
    science_dirs = [os.path.dirname(x) for x in data_paths['lambda']]
    science_paths = {'cube':[]}
    #science_paths = []
    
    target_names = []
    target_dates = []
    target_epochs = []
    target_id = []
    problem_stars = []
    nframes = np.array((),dtype=int)
    data = None
    
    ##reading header information from input cubes for later sorting + removes
    ##paths whose files cannot be found + finds path of science cube
    for path in data_paths['cube']:
        if os.path.dirname(path) in science_dirs:
            science_paths['cube'].append(path)
        try:
            hdr_tmp = read_file(fits.getheader, path)
            if hdr_tmp['OBJECT'] in non_stars or hdr_tmp['DATE-OBS'] in bad_centering:
                problem_stars.append(path)
                continue
            
            try:
                waffle_on = 'CENTER' in hdr_tmp['ESO DPR TYPE']
            except KeyError:
                waffle_on = False
            if hdr_tmp['DATE-OBS'] == '2019-03-23T09:20:29.73': 
                waffle_on = True
                
            if waffle_on:
                if path in science_paths['cube']:
                    raise Exception('> Currently cannot handle science targets with waffle on')
                else:
                    problem_stars.append(path)
                    continue
                
            sid, data = star_info(hdr_tmp['RA'], hdr_tmp['Dec'], known_signal, data, 
                                  hdr_tmp['OBJECT'], hdr_tmp['ESO OBS START'], columns)
            target_names.append(sid)
            target_dates.append(hdr_tmp['DATE'])
            target_epochs.append(hdr_tmp['ESO OBS START'])
            target_id.append(hdr_tmp['OBJECT']+' '+hdr_tmp['ARCFILE']) 
            nframes = np.append(nframes,hdr_tmp['NAXIS3'])
                
        except FileNotFoundError:
            if path in science_paths['cube']:
                raise Exception('> Unable to find science target IRD_SCIENCE_REDUCED_MASTER_CUBE file: {0:s}.',format(path))
            print('> Warning: Could not find data cube file:', path)
            problem_stars.append(path)
    
    data = data.set_index('simbad main identifier')
    cube_paths = [x for x in data_paths['cube'] if x not in problem_stars]
    science_paths['cube'] += match_lambda(cube_paths, data_paths['lambda'], target_epochs, target_dates, nframes)
    
    frame_paths = match_frame_select(data_paths['frame'], target_epochs, nframes)
           
    isci = np.zeros(len(science_paths['cube']),dtype=int)
    for i,sp in enumerate(science_paths['cube']):
        isci[i] = cube_paths.index(sp)
    science_paths['frame'] = [frame_paths[x] for x in isci]
    science_paths['simbad_id'] = [target_names[x] for x in isci]
    science_id = [target_id[x] for x in isci]
   
    ##remove duplicate reductions of the same observation
    problem_stars += remove_dupes(target_id, science_id, isci, nframes, target_dates, cube_paths)
    
    ##science targets with known disks/companions/visible binaries, to keep in ref target
    ##list only if science cube self-correlation is being calculated 
    ##(flagged to skip as a ref cube for all other science targets)
    sci_has_signal = []
    if USE_SCI == True:
        sci_has_signal = remove_known_signal(science_paths['simbad_id'], data, science_paths['cube'])
        print(len(sci_has_signal), "science targets have known disks/companions")
        
    ##flag other observations of the science target
    problem_stars_tmp, skip_sci_target = science_dupes(isci, science_paths['cube'], target_names, 
                                                       target_epochs, cube_paths, sci_has_signal, USE_SCI)   
    if len(problem_stars_tmp) > 0: problem_stars += problem_stars_tmp
    nrm=len(problem_stars)
    ##remove ref targets with known disks/companions/visible binaries
    problem_stars += remove_known_signal(target_names, data, cube_paths, sci_has_signal)
    print(len(problem_stars)-nrm, "reference targets removed for having known disks/companions")
    
    ref_paths = {'cube': [x for x in cube_paths if x not in problem_stars],
                 'frame': [x for i,x in enumerate(frame_paths) if cube_paths[i] not in problem_stars],
                 'simbad_id': [x for i,x in enumerate(target_names) if cube_paths[i] not in problem_stars]}
    ref_paths['parang'] = match_convert(ref_paths['cube'], data_paths['parang'])
    ref_paths['time'] = match_convert(ref_paths['cube'], data_paths['time'])
    
    science_paths['parang'] = match_convert(science_paths['cube'], data_paths['parang'])
    science_paths['time'] = match_convert(science_paths['cube'], data_paths['time'])
    
    return science_paths, ref_paths, skip_sci_target, data_paths['lambda'][0], data, columns