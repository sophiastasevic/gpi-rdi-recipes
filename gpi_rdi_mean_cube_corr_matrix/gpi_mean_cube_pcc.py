#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 10:36:07 2025

@author: sstas
"""
import argparse
import warnings
import numpy as np
from skimage.draw import disk
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning

from file_sorter import get_paths, read_file

warnings.simplefilter('ignore', category=AstropyWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

## initialise header
def init_header(ncubes, path_tmp, r_in, r_out, frm_paths):
    hdr = fits.Header()
    hdr_tmp = read_file(fits.getheader, path_tmp)
    for k in ['PIXTOARC', 'ESO INS COMB IFLT', 'ESO INS2 COMB IFS', 'WL_STACK']:
        hdr[k] = (hdr_tmp[k], hdr_tmp.comments[k])
    
    hdr['FR_STACK'] = ("mean", "method used to collapse the cube along the temporal axis.")
    hdr['NCUBES'] = ncubes
    hdr['CORR_IN'] = (r_in, 'Inner radius of annulus where correlation was computed in px.')
    hdr['CORR_OUT'] = (r_out, 'Outer radius of annulus where correlation was computed in px.')
    
    ## if number of non-'na' frame paths is more than 0, sel_vect = True
    sel_vect = len(frm_paths[frm_paths!='na']) > 0
    hdr['SEL_VECT'] = (sel_vect, "Frame selection vector applied to cubes where available.")
    
    hdr.set('', '---------------------------')
    return hdr
    
def update_header(hdr, i, target_row):
    obj, epoch, sid = target_row[['object','date-obs','simbad main identifier']]
    
    nb_str = '{0:04d}'.format(i)
    hdr['OBJ_'+nb_str] = obj
    hdr['OBS_'+nb_str] = epoch
    hdr['SID_'+nb_str] = sid
    
## create boolean mask for annulus within which correlation is calculated
def create_mask(size, r_in, r_out):
    cxy=(size//2,size//2)
    mask_in=disk(cxy,r_in,shape=(size,size))
    mask_out=disk(cxy,r_out,shape=(size,size))
    mask=np.full((size,size),False)
    mask[mask_out]=True
    mask[mask_in]=False

    return mask

## apply mask to frame and subtract mean    
def mask_frame(frame, mask):
    masked_frame = frame[mask]
    return masked_frame-np.nanmean(masked_frame)

## get indices of data cube frames to keep from frame selection vector if file exists, 
## otherwise use all frames     
def selection_vector(frame_path, nframes_0):
    if frame_path != 'na': 
        select_frames = read_file(fits.getdata, frame_path)
        ## sometimes there are two columns, 0: harsher selection, 1: soft selection
        if len(select_frames.shape)>1: 
            select_frames = select_frames[:,0]
        select_frames = np.where(select_frames==1.)[0]
    else: ## no frame selection vector
        select_frames = np.arange(0,nframes_0)
        
    return select_frames

## mean combine cube along temporal dimention and then apply mask to keep
## only the xy pixels within the pcc calculation region as a 1d array
def format_cube(cube, r_in, r_out):
    cube_stack = np.nanmean(cube, axis=0)
    nx, ny = cube_stack.shape
    mask = create_mask(nx, r_in, r_out)
        
    return mask_frame(cube_stack, mask)

## calculate Pearson correlation coefficient between stacked cubes
def pcc(frames, nframe):
    corr_matrix = np.ones((nframe, nframe))
    for i in range(nframe):
        ## diagonally symmetric so only compute one half of the diagonal
        for j in range(nframe-1,i,-1):
            if i==j: ## same frame, correlation == 1
                continue
            else:
                corr_matrix[i,j] = np.corrcoef(frames[i], frames[j])[0,1]
                corr_matrix[j,i] = corr_matrix[i,j] ## corresponding diagonal
    return corr_matrix
    
#%%
if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('sof', help='name of sof file', type=str)
    parser.add_argument('--r_in',help='Inner radius of correlation annulus.', type=int, default=15)
    parser.add_argument('--r_out',help='Outer radius of correlation annulus.', type=int, default=40)
    
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
    
    ## frame types to read in and minimum number of inputs required
    data_names = {'cube':('GPI_REDUCED_COLLAPSED_MASTER_CUBE', 2),
                  'frame':('GPI_FRAME_SELECTION_VECTOR', 0)}
    
    print("..Getting cube information..")
    cube_paths, frame_select_paths, target_data = get_paths(sofname, data_names)
    
    ncube = len(cube_paths)
    
    hdr = init_header(ncube, cube_paths[-1], r_in, r_out, np.unique(frame_select_paths))
    
    frames = []
    nframe_cube = np.zeros(ncube, dtype=int)
    
    print("..Reading in data cubes..")
    
    for i, (path, frm_path) in enumerate(zip(cube_paths, frame_select_paths)):
        target_row = target_data.loc[path]
        update_header(hdr, i, target_row)
        
        ## read in wl collapsed cube
        cube_tmp = read_file(fits.getdata, path)
        
        select_frames = selection_vector(frm_path, cube_tmp.shape[0]) 
        nframe_cube[i] = len(select_frames) ## for boolean select_frames: np.count_nonzero(select_frames)
        
        ## stack cube along frame axis and apply mask to return a 1d array of the 
        ## pcc calculation region
        stack_frame = format_cube(cube_tmp[select_frames], r_in, r_out)
        frames.append(stack_frame)
        
    target_data['nframe_keep'] = nframe_cube
     
    print("..Saving target information to CSV..")
    target_data.to_csv("target_data.csv")
    
    print("..Calculating correlation between %d stacked cubes.." % ncube)
    
    ## go from a list of 1d arrays to a 2d array of size (ncube, npixel_mask)
    frames = np.stack(frames)
    
    ## calculate PCC between each "frame"
    corr_matrix = pcc(frames, ncube)
    
    print("..Saving correlation matrix to FITS..")
    
    hdu = fits.PrimaryHDU(data=corr_matrix, header=hdr)
    hdu.writeto("mean_cube_pcc_matrix.fits")
    