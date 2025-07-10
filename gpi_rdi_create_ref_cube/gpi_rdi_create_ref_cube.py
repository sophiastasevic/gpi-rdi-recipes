#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 16:17:30 2025

@author: sstas
"""
import os
import argparse
import warnings
import numpy as np
import pandas as pd
from skimage.draw import disk
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning

from data_handler import read_file, read_sof, format_df

warnings.simplefilter('ignore', category=AstropyWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

"""
---------------------------------- FUNCTIONS ----------------------------------
"""

def fix_header(path):
    """
    Read in and fix astropy header (only needed if header is being used for a saved fits file).
    """
    hdul = read_file(fits.open, path)
    hdul.verify('fix')
    
    return hdul[0].header


def complete_header(science_hdr, ncorr, nsample, r_in, r_out):
    """
    Add input parameters to cube header.
    
    Parameters
    ----------
    science_hdr : Header 
        Header of science cube.
    ncorr : int 
        Number of reference frames selected for each science frame.
    nsample : int
        Number of preselected frames.
    r_in : int
        Inner radius of annulus used to calculate correlation.
    r_out : int
        Outer radius of annulus used to calculate correlation.
    """
    science_hdr['USE_SCI'] = (False, 'Science cube included in reference cube list')
    science_hdr['PER_FRM'] = (True, 'Reference library built for each science frame')
    
    science_hdr['NCORR'] = (ncorr, 'Number of best correlated frames selected for reference library')
    science_hdr['NPARAM'] = (nsample, 'Number of preselected frames with best cube correlation')
    science_hdr['CORR_IN'] = (r_in, 'Inner radius of annulus where correlation was computed in px')
    science_hdr['CORR_OUT'] = (r_out, 'Outer radius of annulus where correlation was computed in px')
    

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
    shape = [science_hdr['NAXIS{0:d}'.format(x)] for x in [4,3,2,1]]
    print('> science_cube.shape =', shape)
    science_hdr['SEL_VECT'] = (False, "Frame selection vector applied to science cube.")
    if nframe != shape[1]:
        print('> No. frames after applying frame selection vector =', nframe)
        science_hdr['SEL_VECT'] = True
    else:
        print('> Science cube does not have a frame selection vector')
#%%
"""
Cube formatting
"""   
 
def create_mask(size, r_in, r_out):
    """
    Create boolean mask for annulus within which correlation is calculated.

    Parameters
    ----------
    size : int
        Length of frame x-axis == y-axis
    r_in : int
        Inner radius of annulus
    r_out : int
        Outer radius of annulus
        
    Returns
    -------
    mask : Array of bool
        Mask for correlation calculation region
    """
    
    cxy=(size//2,size//2)
    mask_in=disk(cxy,r_in,shape=(size,size))
    mask_out=disk(cxy,r_out,shape=(size,size))
    mask=np.full((size,size),False)
    mask[mask_out]=True
    mask[mask_in]=False

    return mask

def mask_frame(frame, mask):
    """
    Apply mask to frame and subtract the mean. 
    """
    masked_frame = frame[mask]
    return masked_frame-np.nanmean(masked_frame)

def format_cube(cube, mask, select_frames=None):
    """
    Apply frame selection vector to cube and mask each frame to keep only region
    to calculate PCC for.
    
    Parameters
    ----------
    cube : Array of float
        3D wavelength stacked data cube
    mask : Array of bool
        Mask for correlation calculation region
    select_frames : Array of int, optional
        Frame selection vector of cubes

    Returns
    -------
    masked_cube : Array of float
        2D array of size (nframe, npixels)
    """
    
    if select_frames is not None:
        cube = cube[select_frames]
    
    nframe = cube.shape[0]
    masked_cube = np.zeros((nframe, np.count_nonzero(mask)))
    for i in range(nframe):
        masked_cube[i] = mask_frame(cube[i], mask)    
    
    return masked_cube


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
        ## sometimes there are two columns, 0: harsher selection, 1: soft selection
        if len(select_frames.shape)>1: 
            select_frames = select_frames[:,0]
        select_frames = np.where(select_frames==1.)[0]
    else: ## no frame selection vector
        select_frames = np.arange(0,nframes_0)
        
    return select_frames

#%%
"""
Reference selection
"""

def preselect_cubes(cube_corr, nsample, nframes):
    """
    Identify reference cubes with the best correlation to the science cube.

    Parameters
    ----------
    cube_corr : Array of float
        1D correlation between science cube and reference cubes
    nsample : int
        Number of frames to preselect
    nframes : Array of int
        Frame count of each reference cube

    Returns
    -------
    Array of int
        Index position of preselected cubes
    """
    
    ## index of highest absolute correlation values
    cube_sort = np.argsort(-np.abs(cube_corr))
    
    ## select number of highest correlated cubes that have at least nsample frames between them
    frm_count = 0
    for i, index in enumerate(cube_sort):
        frm_count += nframes[index]
        
        if frm_count >= nsample:
            print("> %d reference frames preselected from %d targets" % (frm_count,i+1))
            
            return cube_sort[:i+1]


def print_ref_correlation(ref_corr):
    """
    Print correlation statistics of reference cube
    """
    ## 0:median, 1:min, 2:max, 3:lower quartile, 4:upper quartile
    names = ['median','min','max','lower quartile','upper quartile']
    stats = np.concatenate((np.stack((np.median(ref_corr), np.min(ref_corr), np.max(ref_corr))),
                            np.quantile(ref_corr,(0.25,0.75))))
    stats_str = ', '.join(['{0:s} = {1:.3f}'.format(x,y) for x,y in zip(names,stats)])
    print('> Reference library correlation:', stats_str)
    
        
def score_frames(corr_matrix, ncorr):
    """
    Select best correlated frames to build reference cube for each science frame.

    Parameters
    ----------
    corr_matrix : Array of float
        Pearson correlation coefficient calculated between science and reference frames.
    ncorr : int
        Number of best correlated reference frames to select for each science frame.
    
    Returns
    -------
    ref select : Array of int
        Index position of selected best correlated reference frames.
    """
    
    sci_nframe, corr_nframe = corr_matrix.shape
    
    ## index of highest absolute correlation values
    ref_select = np.argsort(-np.abs(corr_matrix))[...,:ncorr]
    
    print_ref_correlation(np.take_along_axis(corr_matrix, ref_select, axis=-1))
    
    return ref_select


def sort_frame_list(ref_select, corr_frame_vect):
    """
    Convert reference frame index from position in correlation matrix to position of cube in ref cube
    list and position of frame in cube.

    Parameters
    ----------
    ref_select : Array of int
        Index of selected reference frames
    corr_frame_vect : Series
        Index reference cubes within list of targets and frame within cube, corresponding to each corr_matrix index.
    
    Returns
    -------
    ref_frame_vect : Series
        Index positions of selected reference targets in cube list and selected frames in the cube.
    ref_frame_select : Array of int
        Reference frame selection vector of full cube for each science frame and wavelength channel.
    """
    ## set of reference frames for all science frames     
    frame_index = list(set(ref_select.flatten()))
    frame_index.sort()
     
    sci_nframe = ref_select.shape[0]
    
    ## frame selection vector specifying which reference frames belong to which science frame's
    ## reference library (avoid saving separate ref cube for each one)
    ref_frame_select = np.zeros((sci_nframe, len(frame_index)), dtype=int)
    for isci in range(sci_nframe):
        for i,index in enumerate(frame_index):
            if index in ref_select[isci]:
                ref_frame_select[isci,i] = 1
    
    return corr_frame_vect.iloc[frame_index], ref_frame_select

#%%
"""
Make reference cube
"""

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
        for i, f in enumerate(frms):
            if i == 0:
                frm_str = str(f)
            elif f-frms[i-1]==1 and (i == len(frms)-1 or frms[i+1]-f>1):
                frm_str+="-{0:d}".format(f)
            elif f-frms[i-1]>1:
                frm_str+=",{0:d}".format(f)
        return frm_str
                
    ref_hdr['OBJ_{0:04d}'.format(i)] = ref_row['object']
    ref_hdr['OBS_{0:04d}'.format(i)] = ref_row['date-obs']
    ref_hdr['N_{0:04d}'.format(i)] = len(frame_index)
    ref_hdr['FRM_{0:04d}'.format(i)] = make_frame_string(frame_index)
    if i == 0:
        ref_hdr.comments['N_{0:04d}'.format(i)] = 'Number of frames used from this cube'
        ref_hdr.comments['FRM_{0:04d}'.format(i)] = 'Index of selected frames in original cube'
        

def make_ref_cube(cube_paths, ref_frame_vect, target_data, ref_hdr):
    """
    Read in cube for selected frames and construct the reference library.

    Parameters
    ----------
    cube_paths : list of str
        Preselected reference cube paths
    ref_frame_vect : Series
        Index positions of selected reference targets in cube_paths and selected frames in the cube.
    target_data : DataFrame
        Reference target information.
    ref_hdr : Header
        Initial header for reference cube.
    
    Returns
    -------
    ref_cube : Array of float
        Reference frame cube.
    ref_hdr : Header
        Final header for reference cube.
    """
    ref_index = list(ref_frame_vect.index.unique())
    ref_index.sort()
    ref_path = [x for i,x in enumerate(cube_paths) if i in ref_index]
    ref_cube = []
    
    for i, (refi, path) in enumerate(zip(ref_index,ref_path)):
        frame_index = ref_frame_vect.loc[refi]
        try: ## path SHOULD exist, but just in case it doesn't, we want to move to the next cube
            cube = read_file(fits.getdata, path)
            
            ref_cube.append(cube[frame_index])
            add_header_ref_info(target_data.loc[path], i, frame_index, ref_hdr)
        
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
Frame-to-frame correlation
"""


"""
---------------------------------- MAIN CODE ----------------------------------
"""
if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('sof', help='name of sof file', type=str)
    parser.add_argument('--r_in',help='Inner radius of correlation annulus.', type=int, default=15)
    parser.add_argument('--r_out',help='Outer radius of correlation annulus.', type=int, default=40)
    parser.add_argument('--nsample', help='No. frames to preselect from best correlated cubes.', type=int, default=1000)
    parser.add_argument('--ncorr', help='No. reference frames to select for each science frame.', type=int, default=500)
        
    args = parser.parse_args()
    
    ## sof file path
    sofname = args.sof
    
    ## --inner/outer radius for correlation comparison annulus
    r_in = args.r_in
    r_out = args.r_out
    
    if r_out <= 10:
        r_out = 10
        print('[Warning] outer radius too small. Setting to 10px')
    if r_out <= r_in:
        r_in = r_out-1
        print('[Warning] r_out <= r_in. Inner radius set to {0:d}'.format(r_out-1))
        
    ## --nsample of preselected frames
    nsample = args.nsample
    
    ## --ncorr of reference frames
    ncorr = args.ncorr
    
    """
    Initial data read-in
    """
    ## frame types to read in and minimum number of inputs required
    data_names = {'science':('GPI_REDUCED_COLLAPSED_MASTER_CUBE', 1),
                  'frame':('GPI_FRAME_SELECTION_VECTOR', 0),
                  'corr':('GPI_CORR_MATRIX', 1),
                  'corr_data':('GPI_REFERENCE_TARGET_DATA', 1)}
    
    paths = read_sof(sofname, data_names)
    
    science_hdr = fix_header(paths['science'])
    size = science_hdr['NAXIS2']
    
    ## data frame containing paths of each GPI cube and frame selection vector and index of science cube
    target_data, isci = format_df(paths, science_hdr['DATE-OBS'])
    
    ## mean cube correlation between science cube and all GPI cubes
    science_cube_corr = read_file(fits.getdata, paths['corr'])[isci]
    
    ## mask non reference cubes
    mask_df = target_data[['binary','planet','disk','bad_path']].any(axis='columns').to_numpy()
    mask_df[isci] = True
    
    science_cube_corr[mask_df] = np.nan
    
    """
    Cube preselection
    """
    ## get best correlated cubes
    cube_select = preselect_cubes(science_cube_corr, nsample, target_data['nframe_keep'].to_numpy())
    cube_paths, frame_select_paths = target_data.iloc[cube_select,['path','frame_path']]
    
    """
    Prepare science cube and header
    """
    
    science_cube = read_file(fits.getdata, paths['science'])
    select_frames = selection_vector(paths['frame'], size) 
    
    print_cube_info(science_hdr, np.cout_nonzero(select_frames))
    complete_header(science_hdr, ncorr, nsample, r_in, r_out)
    
    mask = create_mask(size, r_in, r_out)
    
    science_frames = format_cube(science_cube, mask, select_frames)
    
    """
    Frame-to-frame correlation
    """
    
    corr_matrix = []
    corr_frame_vect = []
    
    print("..Calculating frame-to-frame correlation..")
    
    for i, (path, frm_path) in enumerate(zip(cube_paths, frame_select_paths)): 
        cube_tmp = read_file(fits.getdata, path)
        select_frames = selection_vector(frm_path, cube_tmp.shape[0])   
        
        ## save index of cube, i, and index of kept frames
        corr_frame_vect.append(np.stack((np.full_like(select_frames,i), select_frames)))
        
        """
        Apply frame selection and mask to ref cube
        
        Calculate PCC between science frames and ref frames
        
        Append PCC matrix to corr_matrix
        
        """
    
    ## concatenate frame_vect to get a 2d array where col 0 == index of the cube in the
    ## path list, and col 1 == frame index of frame selection vector within the cube
    corr_frame_vect = np.concatenate(corr_frame_vect, axis=1)
    corr_frame_vect = pd.Series(data=corr_frame_vect[1], index=corr_frame_vect[0], name='frame_index')   
    
    corr_matrix = np.concatenate(corr_matrix, axis=1)
    
    """
    Build reference library
    """
    
    ## index of best correlated frames in the matrix
    ref_select = score_frames(corr_matrix, ncorr)
    
    ## cube and frame index of best correlated frames, and science-ref frame match array
    ref_frame_vect, ref_frame_select = sort_frame_list(ref_select, corr_frame_vect)  
    
    ## build reference cube
    ref_cube, ref_hdr = make_ref_cube(cube_paths, ref_frame_vect, target_data.set_index('path'), science_hdr)
    
    hdu = fits.PrimaryHDU(data=ref_cube, header=ref_hdr)
    hdu.writeto('reference_cube.fits')
    
    frame_hdu = fits.PrimaryHDU(data=ref_frame_select, header=ref_hdr)
    frame_hdu.writeto('frame_selection_vector.fits')
