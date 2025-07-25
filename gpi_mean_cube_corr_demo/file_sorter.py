#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reads in GPI collapsed cube headers, queries the star-server to get the
target stellar parameters, and matches the frame selection vectors to the
correct cube.

"""
import warnings
import sys, os, re
import subprocess
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
from retry_handler import RetryHandler

import simbad_star_query as simbadtool

warnings.simplefilter('ignore', category=AstropyWarning)

## prevents calls to print when calling a function using 'with HiddenPrints():'
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        
save_header = ['OBJECT', 'DATE-OBS', 'RA', 'DEC', 'NAXIS3']
save_star = {'flux g':'simbad_FLUX_G', 'flux h':'simbad_FLUX_H', 'flux k':'simbad_FLUX_K', 
             'spectral type':'simbad_SP_TYPE', 'simbad main identifier':'simbad_MAIN_ID'}
        
"""
---------------------------------- FUNCTIONS ----------------------------------
"""
def read_file(func, path, **kwargs):
    if "summer" in path:
        return RetryHandler(func).run(path, **kwargs)
    else:
        return func(path, **kwargs)
     

def get_constants_paths(fnames, ftypes):
    signal_paths = fnames[np.where(ftypes=="GPI_RDI_REFERENCE_TARGET_FLAG")[0]]
    return signal_paths


def get_sof_paths(fnames, ftypes, frame, nmin):
    path = fnames[np.where(ftypes==frame)[0]]
    if len(path) < nmin:
        raise Exception('[Error] Input must contain at least', nmin, frame)
    
    path.sort()
    path = np.array([os.path.join(os.getcwd(), p) for p in path])
        
    return path[::-1] ## returns most recently reduced first

def read_sof(sof, data_names):
    data = np.loadtxt(sof, dtype=str)
    fnames = data[:,0] #file names
    ftypes = data[:,1] #file data types

    paths = {}
    for var in data_names.keys():
        paths[var] = get_sof_paths(fnames, ftypes, *data_names[var])
    
    signal_paths = get_constants_paths(fnames, ftypes)
    return paths, signal_paths


## load list of objects with a known companion, disk, or visible binary
def get_companion_disk_flag(signal_paths):
    paths = {}
    for k in ["disk", "planet", "binary"]:
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
    

## query COBREX star service using coords + return output csv as a pandas dataframe
def query_star(ra, dec):
    filename = 'data/star_data.csv'
    url = "http://cobrex-dc.osug.fr:8080/star-server/api/fcoo/csv/d%20{0:f}%20{1:f}".format(ra,dec)
    
    request_star_str = ['wget', '-O', filename, url]
    output,error = subprocess.Popen(request_star_str, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()
    df = pd.read_csv(filename, sep=';', skiprows=0, skipfooter=0, parse_dates=True, quoting=3)
    
    binary = False
    ## more than one star identified, flag as binary if distance between them is
    ## smaller than half the IRDIS FOV (assumes one star is centered)
    if df.shape[0] > 1:
        ## only difference between first two objects (if > 2) since this is the smallest sep
        sep = df['dist[deg]'].diff()[1]*3600 ##arcsec
        if sep < (512*0.01225):
            binary = True
            if pd.isna(df.loc[0,'flux g']) and pd.notna(df.loc[1,'flux g']):
                return df.loc[1].to_dict(), binary
    
    return df.loc[0].to_dict(), binary


clean_str = lambda name: re.sub(r'\s+',' ',name.strip())
## query COBREX star service for object by coord, if this fails, use simbad query 
def star_info(name, date, ra, dec, known_signal, star_data):
    query, binary = query_star(ra, dec)
    if query['sphere id'] == -1: ## star server query failed
        with HiddenPrints(): sim = simbadtool.get_star_info(name, date, ra, dec)
        if sim is None: ## simbad query failed, try again without object name
            with HiddenPrints(): sim = simbadtool.get_star_info(None, date, ra, dec)
            
        for k in save_star.keys():
            query[k] = sim[save_star[k]]
    else:
        query['spectral type'] = query['spectral type'].replace(' ~','')
    query['simbad main identifier'] = clean_str(query['simbad main identifier'])
        
    sid = query['simbad main identifier']
    ## first time querying this star, so get the disk/binary/companion info
    if star_data is None or sid not in list(star_data['simbad main identifier']):
        ## known binary/planet list have names without the "A"/"B"/etc/ identifier, so remove from sid
        sid_tmp = sid
        while sid_tmp[-1] in ['A','B','C','b']:
            sid_tmp = sid_tmp[:-1].strip()
        
        for col in known_signal.keys():
            if sid_tmp in known_signal[col] or clean_str(sid_tmp) in known_signal[col]:
                query[col] = True
            elif col == 'binary': ## star name isn't in known binary list but might have other flags
                query[col] = binary ## binary flag returned by star server query
            else:
                query[col] = False
                
        ## return either new star database, or concatenated databases
        query_data = pd.Series(query).to_frame().T.set_index('simbad main identifier', drop=False)
        if star_data is None:
            return sid, query_data
        else:
            return sid, pd.concat([star_data, query_data])
    ## star already in database so don't need to add again
    else:
        return sid, star_data       

## reading header information from input frame selection vectors to match to
## data cubes + removes paths with missing files
def match_frame_select(data_paths, target_epochs, nframes):
    frame_paths_tmp = []
    frame_epochs = []
    frame_nframe = []
    for frm_path in data_paths['frame']:
        try:
            frame_hdr = read_file(fits.getheader, frm_path)
            
            frame_paths_tmp.append(frm_path)
            frame_epochs.append(frame_hdr['DATE-OBS'])
            frame_nframe.append(frame_hdr['NAXIS{0:d}'.format(frame_hdr['NAXIS'])])
        
        except FileNotFoundError:
            print('> Warning: Could not find GPI_FRAME_SELECTION_VECTOR file:', frm_path)
    
    ## matches each data cube to its frame selection vector
    frame_paths = []
    for ei, epoch in enumerate(target_epochs):
        framei = [i for i,x in enumerate(frame_epochs) if x==epoch and frame_nframe[i]==nframes[ei]]
        if len(framei)==0:
            frame_paths.append('na')
        else:
            frame_paths.append(frame_paths_tmp[framei[-1]]) ## -1 is most recent file (if multiple)
        
    return frame_paths
        
## reading in + sorting file paths read from sof file
def get_paths(sofname, data_names):
    data_paths, signal_paths = read_sof(sofname, data_names)
    ## list of objects observed by SPHERE that are not stars
    known_signal = get_companion_disk_flag(signal_paths)
    
    target_data = {k.lower():[] for k in save_header+list(save_star.keys())+['binary','planet','disk']}
    star_data = None
    cube_paths = []
    
    ## reading header information from input cubes to check if file exists and identify star
    for path in data_paths['cube']:
        try:
            hdr_tmp = read_file(fits.getheader, path)
            if hdr_tmp['DATE-OBS'] in target_data['date-obs']:
                print('> Skipping duplicate reduction of %s %s observation' % (hdr_tmp['OBJECT'], hdr_tmp['DATE-OBS']))
                continue

            sid, star_data = star_info(*[hdr_tmp[k] for k in save_header[:-1]], known_signal, star_data)
            
            cube_paths.append(path)
            
            ## save header and star info to the dict
            for k in list(save_star.keys())+['binary','planet','disk']:
                target_data[k].append(star_data.loc[sid, k])
            
            for k in save_header:
                target_data[k.lower()].append(hdr_tmp[k])
                   
        except FileNotFoundError:
            print('> Warning: Could not find GPI_REDUCED_COLLAPSED_MASTER_CUBE file:', path)
    
    df = pd.DataFrame(target_data, index=pd.Index(cube_paths, name='path'))
    df.rename(columns={'naxis3':'nframes'}, inplace=True)
    
    print("..Matching GPI_FRAME_SELECTION_VECTOR paths ..")
    ## match paths to frame selection vectors
    frame_select_paths = match_frame_select(data_paths, df['date-obs'], df['nframes'])
    
    df['frame_path'] = frame_select_paths
    
    return cube_paths, frame_select_paths, df