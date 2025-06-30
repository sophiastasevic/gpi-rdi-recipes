#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@AUTHOR: Sophia Stasevic
@DATE: 28/05/25
@CONTACT: sophia.stasevic@obspm.fr

Create a reference library for a specific GPI_SCIENCE_REDUCED_MASTER_CUBE to perform RDI reduction.
Reference library selection steps:
    
    1)  Find eligible reference cubes from list of imported GPI_SCIENCE_REDUCED_MASTER_CUBE by removing
        observations of targets with disks/binaries/companions or taken with satellite spots.
    
    2)  Get target and frame information, either from the cube header, COBREX star-server query, or
        COBREX asm-server query. [Option to include previous recipe star or asm data output as input]
    
    3a) [MIX or single parameter] Either preselect frames from the eligable reference cubes based on 
        how well they match the science cube or frame parameter, either using only one parameter 
        [parameter name], or every parameter except for NDIT [MIX]. 
         
         Parameters include: 
          - SEEING
          - TAU0 
          - LWE     (30m wind speed)
          - WDH     (200mbar wind speed + direction) 
          - ELEV    (elevation of target)
          - DIT     (detector integration time)
          - NDIT    (number of integrations)
          - EPOCH 
          - MG      (G-band magnitude), 
          - MFILT   (H- or K-band magnitude)
          - SPT     (spectral type)
                
     b) [PCC, RAND] OR consider frames from all eligible cubes.
    
    4a) [RAND] Randomly select frames from the eligable frame list to create reference cube.
    
     b) Else, get the Pearson Correlation Coefficient (PCC) between the science and (preselected) 
        reference cube frames within a defined annulus (either query COBREX correlation-server or
        calculate) and either:
             
             i) [PCC, single parameter] Select the best correlated frames.
    
            ii) [MIX] Select a subset of best correlated frames from each parameter preselection and
                concatenate for final selection.
        Note: Correlation can be calculated for one or both wavelength channels, leading to different
              libraries for each channel. Reference libraries are constructed for each science frame.
                
    5) Read in selected reference frames create reference cube using cropped frames.
    
Input data:
    GPI_SCIENCE_LAMBDA_INFO [1] | lambda file of science cube
    GPI_SCIENCE_REDUCED_MASTER_CUBE | science and reference cubes
    GPI_FRAME_SELECTION_VECTOR [optional] | frame selection vectors of cubes
    GPI_SCIENCE_PARA_ROTATION_CUBE | parralactic angles of cube frames
    GPI_TIMESTAMP | observing time of cube frames
    GPI_REFERENCE_TABLE [optional] | fits table with target header and star information
    GPI_REFERENCE_FRAME_DATA [optional] | csv with asm and frame data
    
    GPI_RDI_TABLE_COLUMN_META [1] | static csv file containing for a list of parameters their: 
                                        GPI_REFERENCE_TABLE column name [index of csv]
                                        GPI_SCIENCE_REDUCED_MASTER_CUBE header name (if appl.)
                                        COBREX star-server query name (if appl.)
                                        SIMBAD query name (if appl.) [backup if star-server fails]
                                        data type
                                        unit
                                        data source (i.e., header, star-server query, etc.)
                                        
    GPI_RDI_REFERENCE_TARGET_FLAG [4] | static files containing list of targets that are not stars
                                        OR have disks/binaries/companions 

Input parameters:
    sof | File name of the sof file
    --use_science | Use the science cube as a reference target.
    --reload_framevect | Read in frame selection vector even if cached data is available.
    --param | Parameter to use for reference frame (pre)selection, 
              options: MIX, PCC, RAND, SEEING, TAU0, LWE, WDH, DIT, NDIT, EPOCH, MG, MFILT, SPT
    --param_max | Number of frames to preselect.
    --ncorr | Number of (best correlated) reference frames to select for each science frame.
    --crop | Pixel size to crop reference frames to.
    
    --r_in | Inner radius of correlation annulus.
    --r_out | Outer radius of correlation annulus.
    --lambda_r | Inner and outer radii given as a function of lambda/D. (ONLY IF wl_channels = 0 OR 1)
    --wl_nchannel | Wavelength channel to use. 0=Channel 1, 1=Channel 2, 2=Channels 1 and 2
                                        
Outputs:
    GPI_REFERENCE_CUBE | fits cube of reference frames for all science frames
    GPI_REFERENCE_FRAME_SELECTION_VECTOR | frame selection vector for reference cube
    GPI_REFERENCE_TABLE | fits table with target header and star information of all input cubes
    GPI_REFERENCE_FRAME_DATA [optional] | csv with asm and frame data of eligible reference cubes

Scripts in recipe:
    ird_create_ref_cube.py [this one]
    file_sorter.py | read in sof, match paths of different input data types, call rdi_ref_table.py
    rdi_ref_table.py | query star-server and compile table with star and header info for each target
    query_asm.py | query asm-server using frame timestamps and compile pandas dataframe with frame info
    ref_frame_preselection.py | preselect best matching reference frames based on parameter values
    build_pcc_matrix.py | get frame-to-frame PCC between science and (pre)selected reference targets 
    
"""
import argparse
import warnings
import os
import numpy as np
import pandas as pd

from astropy.io import fits
from astropy import units as u
from astropy.utils.exceptions import AstropyWarning

import file_sorter
from file_sorter import read_file
from ref_frame_preselection import preselection
from frame_correlation import correlation, print_ref_correlation

warnings.simplefilter('ignore', category=AstropyWarning)
D_TEL = 8.2 #m
AO_WID = 20 #px; width of annulus for measuring ao correction radius correlation

## possible parameters to define reference library preselection
PARAM_LIST = ['PCC','RAND','EPOCH','MG','MH','SPT']

round_up = lambda x: int(x) + (x-int(x)>0)
#%%
"""
---------------------------------- FUNCTIONS ----------------------------------
"""   

def fix_header(path):
    """
    Read in and fix astropy header (only needed if header is being used for a saved fits file).
    
    Parameters
    ----------
    path : str 
        Path to fits file.
    """
    hdul = read_file(fits.open, path)
    hdul.verify('fix')
    
    return hdul[0].header


def complete_header(science_hdr, ncorr, param, param_max, r_in, r_out, USE_SCI):
    """
    Add input parameters to cube header.
    
    Parameters
    ----------
    science_hdr : Header 
        Header of science cube.
    ncorr : int 
        Number of reference frames selected for each science frame.
    param : str
        Parameter used for reference frame (pre)selection.
    param_max : int
        Number of preselected frames.
    r_in : int
        Inner radius of annulus used to calculate correlation.
    r_out : int
        Outer radius of annulus used to calculate correlation.
    USE_SCI : bool
        Flag for including science cube as a reference target.
    """
    science_hdr['USE_SCI'] = (USE_SCI, 'Science cube included in reference cube list')
    science_hdr['PER_FRM'] = (False, 'Reference library built for each science frame')
    science_hdr['PARAM'] = (param, 'Parameter used to (pre)select reference library')

    if param != 'RAND':
        science_hdr['NCORR'] = (ncorr, 'Number of best correlated frames selected for reference library')
        science_hdr['NPARAM'] = (param_max, 'Number of preselected frames with best matching PARAM')
        science_hdr['PER_FRM'] = True
        science_hdr['CORR_IN'] = (r_in, 'Inner radius of annulus where correlation was computed in px')
        science_hdr['CORR_OUT'] = (r_out, 'Outer radius of annulus where correlation was computed in px')
        science_hdr['WL_STACK'] = (True, 'Correlation calculated on wavelength collapsed cube')
    else:
        science_hdr['NCORR'] = (ncorr, 'Number of frames selected for reference library')
        

def print_cube_info(science_hdr, nframe):
    """
    Print science cube info + save info to header.

    Parameters
    ----------
    science_hdr : Header 
        Header of science cube.
    nframe : int
        Number of frames in science cube after applying frame selection vector.
    """
    print('\n>> ESO INS COMB ICOR:', science_hdr['ESO INS COMB ICOR'])
    print('>> ESO INS COMB IFLT:', science_hdr['ESO INS COMB IFLT'])  
    
    print('\n------ Science cube ------')
    print('>> OBJECT:', science_hdr['OBJECT'])
    print('>> DATE-OBS:', science_hdr['DATE-OBS'])
    print('>> DATASUM:', science_hdr['DATASUM'])
    print('>> EXPTIME:', science_hdr['EXPTIME'])
    shape = [science_hdr['NAXIS{0:d}'.format(x)] for x in [3,2,1]]
    print('> science_cube.shape =', shape)
    science_hdr['SEL_VECT'] = (False, "Frame selection vector applied to science cube.")
    if nframe != shape[0]:
        print('> No. frames after applying frame selection vector =', nframe)
        science_hdr['SEL_VECT'] = True
    else:
        print('> Science cube does not have a frame selection vector')
          

def selection_vector(frame_path, nframes_0):
    """
    Get indices of data cube frames to keep from frame selection vector if file exists, 
    otherwise all frames are used.   

    Parameters
    ----------
    frame_path : str
        IRD_FRAME_SELECTION_VECTOR path.
    nframes_0 : int
        Original number of frames in corresponding IRD_SCIENCE_REDUCED_MASTER_CUBE.

    Returns
    -------
    select_frames : Array of int
        Index positions of frames to keep in the cube.
    """
    if frame_path != 'na':
        select_frames = read_file(fits.getdata, frame_path)
        if len(select_frames.shape)>1: 
            select_frames = select_frames[:,0]
        select_frames = np.where(select_frames==1.)[0]
    else:
        select_frames = np.arange(0,nframes_0)
        
    return select_frames


def get_frame_data(paths, cubei, target_row):
    """
    Read in frame selection vector.

    Parameters
    ----------
    paths : Dict
        Cube, frame selection vector, parallactic angle, and timestamp paths of the given target.
    cubei : int
        Index position of the target within the list of targets.
    target_row : Table Row
        Row of target in the input target information table
    
    Returns
    -------
    frms : Array of int
        Index position of target within list of targets and its frame selection vector.
    """
    select_frames = selection_vector(paths['frame'], target_row['Cube_Nframes'])      
    
    frms = np.zeros((2,len(select_frames)), dtype=int)
    frms[0] = cubei
    frms[1] = select_frames
    
    return frms


def score_frames(corr_matrix, corr_index, frame_vect, ncorr, best_param, param_max):
    """
    Select best correlated frames to build reference cube for each science frame.

    Parameters
    ----------
    corr_matrix : Array of float
        Pearson correlation coefficient calculated between science and reference frames.
    corr_index : Array of int
        Index positions of reference cubes within list of targets corresponding to the corr_matrix frames.
    frame_vect : Series
        Index positions of reference cubes within list of targets and frame selection vectors for each cube.
    ncorr : int
        Number of best correlated reference frames to select for each science frame.
    best_param : Array of int
        Index position of preselected reference frames within frame_vect.
    param_max : int
        Number of preselected reference frames.
    
    Returns
    -------
    ref_select : Array of int
        Index position of selected best correlated reference frames within frame_vect.
    best_corr_val : Array of float, (return_mask == True)
        Correlation values of selected frames
    mask_corr : Array of bool, (return_mask == True)
        Mask for selected reference frames in corr_matrix
    """
    def match_frame_index(best_cubei, corr_index, frame_tmp):
        """
        Convert index position within cube to index position within corr_matrix
        """
        match=[]
        for refi in best_cubei:
            corr_framei = np.where(corr_index == refi)[0] ## index of ref frames withinin corr_matrix
            cube_framei = frame_tmp.loc[refi] ## selection vector for ref cube
            if cube_framei.ndim == 0: ## make sure an array of ndim==1 is being appended 
                match.append(corr_framei[np.array((cube_framei,))])
            else:
                match.append(corr_framei[cube_framei.to_numpy()]) ## index of selection vector ref frames within corr_matrix
        return np.concatenate(match)

    nchannel, sci_nframe, corr_nframe = corr_matrix.shape
        
    if best_param.ndim == 1: ## only cube based selection
        select_vect = match_frame_index(best_param, corr_index, frame_vect)
        corr_matrix_tmp = corr_matrix[..., select_vect]
        ## convert from cube to frame indexing
        best_param = np.arange(len(frame_vect))[frame_vect.index.isin(best_param)]
        
    else:
        select_vect = np.zeros((nchannel, sci_nframe, param_max), dtype=int)
        for isci in range(sci_nframe):
            frame_tmp = frame_vect.iloc[best_param[isci]]
            best_cubei = frame_tmp.index.unique().to_list()
            select_vect[:,isci] = match_frame_index(best_cubei, corr_index, frame_tmp)
        corr_matrix_tmp = np.take_along_axis(corr_matrix, select_vect, axis=-1)
    
    ## get index highest absolute correlation values within corr_matrix_tmp
    ## matrix is -ve to account for nan values that would be put up front using [::-1]
    best_corri = np.argsort(-np.abs(corr_matrix_tmp))[...,:ncorr]
    best_corr_val = np.take_along_axis(corr_matrix_tmp, best_corri, axis=-1)
    
    ## convert [index in corr_matrix_tmp] to [index in frame_vect], shape = nchannel x sci_nframe x ncorr
    ## best_param needs to be same shape as best_corri to use np.take_along_axis
    best_param_tmp = np.broadcast_to(best_param, corr_matrix_tmp.shape)
    ref_select = np.take_along_axis(best_param_tmp, best_corri, axis=-1)
    
    print_ref_correlation(best_corr_val)    
    return ref_select 


def sort_frame_list(ref_select, frame_vect, sci_nframe, ncorr):
    """
    Convert reference frame index from positional to Series index + make selection vector.

    Parameters
    ----------
    ref_select : Array of int
        DESCRIPTION.
    frame_vect : Series
        Index positions of reference cubes within list of targets and frame selection vectors for each cube.
    sci_nframe : int
        Number of science frames.
    ncorr : int
        Number of reference frames selected for each science frame.

    Returns
    -------
    ref_frame_vect : Series
        Index positions of selected reference targets in ref_path_tmp and selected frames in the cube.
    ref_frame_select : Array of int
        Reference frame selection vector of full cube for each science frame and wavelength channel.
    """
    ## set of reference frames for all science frames     
    frame_index = list(set(ref_select.flatten()))
    frame_index.sort()
    
    if ref_select.shape != (sci_nframe,ncorr):
        ref_select = np.broadcast_to(ref_select, (sci_nframe,ncorr))
        
    ## frame selection vector specifying which reference frames belong to which science frame's/
    ## wavelength nchannel' reference library (avoid saving separate ref cube for each one)
    ref_frame_select = np.zeros((sci_nframe, len(frame_index)), dtype=int)
    for isci in range(sci_nframe):
        for i,index in enumerate(frame_index):
            if index in ref_select[isci]:
                ref_frame_select[isci,i] = 1
    
    return frame_vect.iloc[frame_index], ref_frame_select


def random_selection(frame_vect, ncorr, isci, USE_SCI):
    """
    Random frame selection from all eligible master reference cubes.

    Parameters
    ----------
    frame_vect : Series
        Index positions of reference cubes within list of targets and frame selection vectors for each cube.
    ncorr : int
        Number of reference frames to select.
    nchannel : int
        Number of wavelength channels.
    isci : int
        Index position of science cube in reference target list.
    USE_SCI : bool
        Flag for including science cube as a reference target.

    Returns
    -------
    ref_frame_vect : Series
        Index positions of selected reference targets in ref_path_tmp and selected frames in the cube.
    ref_frame_select : Array of int
        Reference frame selection vector of full cube for each science frame and wavelength channel.
    """
    sci_nframe = frame_vect.loc[isci].size
    if USE_SCI == False: ## remove science data from Series
        frame_vect = frame_vect.copy().drop(index=isci)
    
    ref_frame_vect = frame_vect.sample(n=ncorr)
    ref_frame_select = np.ones((sci_nframe, ncorr), dtype=int)
    
    return ref_frame_vect, ref_frame_select


def add_header_ref_info(ref_row, i, frame_index, ref_hdr):
    """
    Add reference frame information to reference library header.

    Parameters
    ----------
    ref_row : Table Row
        Row of target in the input target information table
    i : int
        Index position of target cube within list of selected targets for the reference cube.
    frame_index : Array of int
        Index positions of selected reference frames within the current cube.
    ref_hdr : Header
        Header for reference cube.
    """
    def make_frame_string(frms):
        """
        Convert frame indices from Array of int to a string
        """
        frms = list(frms)
        frms.sort()
        frm_str = str(frms[0])
        for i, f in enumerate(frms):
            if i == 0:
                frm_str = str(f)
            elif f-frms[i-1]==1 and (i == len(frms)-1 or frms[i+1]-f>1):
                frm_str+="-{0:d}".format(f)
            elif f-frms[i-1]>1:
                frm_str+=",{0:d}".format(f)
        return frm_str
                
    ref_hdr['OBJ_{0:04d}'.format(i)] = ref_row['Object']
    ref_hdr['OBS_{0:04d}'.format(i)] = ref_row['Obs_date']
    ref_hdr['ID_{0:04d}'.format(i)] = ref_row['Cube_ID']
    ref_hdr['N_{0:04d}'.format(i)] = len(frame_index)
    ref_hdr['FRM_{0:04d}'.format(i)] = make_frame_string(frame_index)
    if i == 0:
        ref_hdr.comments['ID_{0:04d}'.format(i)] = 'Unique identifier taken from cube DATASUM header'
        ref_hdr.comments['N_{0:04d}'.format(i)] = 'Number of frames used from this cube'
        ref_hdr.comments['FRM_{0:04d}'.format(i)] = 'Index of selected frames in original cube'
    

def make_ref_cube(ref_path_tmp, ref_frame_vect, target_table, ref_hdr, crop):
    """
    Read in cube for selected frames and construct the reference library.

    Parameters
    ----------
    ref_path_tmp : list of str
        Reference cube paths.
    ref_frame_vect : Series
        Index positions of selected reference targets in ref_path_tmp and selected frames in the cube.
    target_table : Table
        Reference target information table.
    ref_hdr : Header
        Initial header for reference cube.
    crop : int
        Size to crop reference frames to in pixels.

    Returns
    -------
    ref_cube : Array of float
        Reference frame cube.
    ref_hdr : Header
        Final header for reference cube.
    """
    ref_index = list(set(ref_frame_vect.index))
    ref_index.sort()
    ref_path = [x for i,x in enumerate(ref_path_tmp) if i in ref_index]
    ref_cube = []
    
    for i, (refi, path) in enumerate(zip(ref_index,ref_path)):
        frame_index = ref_frame_vect.loc[refi]
        try: ## path SHOULD exist, but just in case it doesn't, we want to move to the next cube
            cube = read_file(fits.getdata, path)
            if crop is not None:
                size = cube.shape[-1]
                if (size+crop)%2:
                    size+=1 ## for accurate centering
                s_min, s_max =  ((size-crop)//2, (size+crop)//2)
                cube = cube[...,s_min:s_max,s_min:s_max]
            
            ref_cube.append(cube[frame_index])
            add_header_ref_info(target_table.loc[path], i, frame_index, ref_hdr)
        
        except FileNotFoundError:
            print('> [Warning] File not found with path: {0:s}, \n\
                  \t.. Continuing to next reference target .. \n'.format(path))
        
    ref_hdr.set('NCUBE', len(ref_cube), 'Total number of reference targets used', before='OBJ_0000')
    ref_hdr.set('NFRAME', len(ref_frame_vect), 'Total number of reference frames', after='NCUBE')
    ref_hdr.set('', '----------------', after='NFRAME')

    print('> Reference library built using {0:d} reference target(s) and {1:d} frame(s).'
          .format(len(ref_cube),len(ref_frame_vect)))
    
    return np.concatenate(ref_cube, axis=1), ref_hdr
#%%           
"""
---------------------------------- MAIN CODE ----------------------------------
"""
if __name__=='__main__':
    """
    Input parameter conversions
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('sof', help='File name of the sof file.', type=str)
    parser.add_argument('--use_science', help='Include the science cube as a reference target (ARDI).', action='store_true')
    ## reference library related arguments:
    parser.add_argument('--param', help='Parameter to use for frame (pre)selection.', type=str, choices=PARAM_LIST, default='MIX')
    parser.add_argument('--param_max', help='No. frames to preselect which closely matching the param of the science frame.', type=int, default=10000)
    parser.add_argument('--ncorr', help='No. reference frames to select for each science frame.', type=int, default=100)
    parser.add_argument('--crop', help='Pixel size to crop reference frames to.', type=int)
    ## correlation related parameters
    parser.add_argument('--r_in',help='Inner radius of correlation annulus.', type=int, default=15)
    parser.add_argument('--r_out',help='Outer radius of correlation annulus.', type=int, default=40)
    
    args = parser.parse_args()
        
    ## sof file path
    sofname = args.sof
    
    ## --use_science flag [True|False]
    USE_SCI = args.use_science
    
    ## --param to use for preselection [see script header for options]
    param = (args.param).strip().upper()
    
    ## --ncorr frame size of reference library
    ncorr = args.ncorr
    
    ## --param_max frame size of preselection
    param_max = args.param_max
    if param_max < ncorr:
        raise Exception('[Error] number of reference frames must be larger than or equal to preselection number.')
    
    ## --crop size of reference frames
    crop = args.crop
    
    ## --inner/outer radius for correlation comparison annulus
    r_in = args.r_in
    r_out = args.r_out
    
    if r_out <= 10:
        r_out = 10
        print('[Warning] outer radius too small. Setting to 10px')
    if r_out <= r_in:
        r_in = r_out-1
        print('[Warning] r_out <= r_in. Inner radius set to {0:d}'.format(r_out-1))
    
    """
    Data paths and target cube tables
    """
    ## data type name in sof file and minimum no. files required
    data_names = {'cube':('GPI_REDUCED_COLLAPSED_MASTER_CUBE', 1+USE_SCI),
                  'frame':('GPI_FRAME_SELECTION_VECTOR', 0),
                  'target_table':('GPI_REFERENCE_TABLE', 0)}
    ## dict of science paths matching lambda input, and master reference cube paths + info table
    ## info row of science target returned in case USE_SCI == False and so not in returned table
    sci_paths, ref_paths, target_table = file_sorter.get_paths(sofname, data_names, USE_SCI)
    
    target_table.remove_indices('Master_ref') ## index that was added in file_sorter.get_paths
    target_table.add_index('Path')
    #%%
    ncube_ref = len(ref_paths['cube'])
    if ncube_ref < 1:
        raise Exception('No eligible reference targets')
    
    ncubes = ncube_ref + (1-USE_SCI) ##no. unique file names
    print('\n> Master reference library contains', ncube_ref, 'reference cubes(s).')
    if USE_SCI:
        print('> Science target included in master reference library.')
        
    """
    Frame information
    """
    
    frame_vect = []
    nframes = np.zeros(ncubes, dtype=int)
    cubei_table = np.zeros_like(nframes) ## make sure paths and table rows are in the same order
    ## trick using dict that appends the science path to the list of paths being enumerated
    ## if USE_SCI == 0 (False), i.e. science path is not already part of ref paths list
    for cubei, path in enumerate(ref_paths['cube']+{0:[sci_paths['cube']],1:[]}[USE_SCI]):
        if path == sci_paths['cube']:
            sci_refi = cubei
            paths = sci_paths
        else:
            paths = {k:ref_paths[k][cubei] for k in ref_paths.keys()}
        frms = get_frame_data(paths, cubei, target_table.loc[path])
        
        frame_vect.append(frms)
        nframes[cubei] = frms.shape[1]  
        cubei_table[cubei] = target_table.loc_indices[path] ## row index of path in the table
    
    target_table = target_table[cubei_table]
    frame_vect = np.concatenate(frame_vect, axis=1)
                         #frame index in cube  #cube index in path list
    frame_vect = pd.Series(data=frame_vect[1], index=frame_vect[0], name='frame_index')
    print("> Master reference library contains {0:d} frames.".format(len(frame_vect.loc[np.arange(ncube_ref)])))
    
    if len(frame_vect) < ncorr:
        raise Exception('[Error] Not enough input frames for requested reference library size.')
    elif len(frame_vect) < param_max:
        raise Exception('[Error] Not enough input frames for requested preselection size.')
    
    """
    Finalising parameters and science header information
    """
    science_hdr = fix_header(sci_paths['cube'])
    sci_select = frame_vect.loc[sci_refi]
    sci_nframe = len(sci_select)
    print_cube_info(science_hdr, sci_nframe)
    
    complete_header(science_hdr, ncorr, param, param_max, r_in, r_out, USE_SCI)

    """
    Preselection and reference cube frame selection
    """
    print('\n.. Creating reference library with best matching {0:s} frames ..'.format(param))
    ## random reference library
    if param == 'RAND':
        ref_frame_vect, ref_frame_select = random_selection(frame_vect, ncorr, sci_refi, USE_SCI)
    else:
        ## Pearson Correlation Coefficient selection using all master reference cubes
        if param == 'PCC':
            best_param = np.arange(ncube_ref)
            science_hdr['NPARAM'] = np.sum(nframes[best_param])
        ## parameter preselection
        else:
            ## ---- Change this module to select based on epoch ----
            best_param = preselection(param, target_table, sci_refi, nframes, ncorr, param_max, USE_SCI)
        
        ## get frame-to-frame correlation of preselected reference cubes
        print(".. Getting frame-to-frame correlation ..")
        corr_matrix, corr_index = correlation(sci_paths['cube'], ref_paths['cube'], target_table, 
                                              frame_vect, best_param, r_in, r_out, sci_select)
        
        ## select best correlated reference frames from the preselection
        ref_select = score_frames(corr_matrix, corr_index, frame_vect, ncorr, best_param, param_max)
            
        ## convert ref_select indices into a Series and create a selection vector for the reference cube
        ref_frame_vect, ref_frame_select = sort_frame_list(ref_select, frame_vect, sci_nframe, ncorr)  
    """
    Build and save reference cubes
    """
    ref_cube, ref_hdr = make_ref_cube(ref_paths['cube'], ref_frame_vect, target_table, science_hdr, crop)
    
    hdu = fits.PrimaryHDU(data=ref_cube, header=ref_hdr)
    hdu.writeto('reference_cube_{0:s}.fits'.format(param))
    
    frame_hdu = fits.PrimaryHDU(data=ref_frame_select, header=ref_hdr)
    frame_hdu.writeto('frame_selection_vector_{0:s}.fits'.format(param))