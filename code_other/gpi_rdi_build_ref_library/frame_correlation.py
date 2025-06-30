#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build matrix of correlation values between science and 
"""
import warnings
import numpy as np

from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
from skimage.draw import disk

from file_sorter import read_file

warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter('ignore', category=AstropyWarning)

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

## read in cube and create mask if not existant, apply mask to frames
def format_cube(path, r_in=None, r_out=None, mask=None, select_frames=None):
    cube = read_file(fits.getdata, path)
    if select_frames is not None:
        cube = cube[select_frames]
    nframe, nx, ny = cube.shape
    if mask is None:
        mask = create_mask(nx, r_in, r_out)
        
    masked_cube = np.zeros((nframe, np.count_nonzero(mask)))
    for i in range(nframe):
        masked_cube[i] = mask_frame(cube[i], mask)
    
    if r_in == None and r_out == None:
        return masked_cube
    else:
        return masked_cube, mask


## calculate frame-to-frame Pearson correlation coefficient between two cubes
def calc_pcc(sci_frames, ref_frames):
    sci_nframe, ref_nframe = sci_frames.shape[0], ref_frames.shape[0]
    corr_mat_tmp = np.zeros((sci_nframe, ref_nframe))
    for i in range(sci_nframe):
        for j in range(ref_nframe):
            corr_mat_tmp[i,j] = np.corrcoef(sci_frames[i], ref_frames[j])[0,1] 
    return corr_mat_tmp

## get Pearson correlation coefficent within annulus between all science cube and reference 
## cube frames by reading in cube and calculating value
def get_pcc(sci_frames, ref_path, sci_id, ref_id, r_in, r_out, mask, 
            sci_nframe0, ref_nframe0, sci_select, ref_select, ref_frames=None):
    ## sci and ref frame size is that of the original cubes, not ones after applying selection vector
    corr_mat = np.full((sci_nframe0, ref_nframe0), np.nan, dtype=np.float32)
    
    ref_frames = format_cube(ref_path, mask=mask, select_frames=ref_select)
    corr_mat[np.ix_(sci_select, ref_select)] = calc_pcc(sci_frames, ref_frames)
    
    return corr_mat[:,sci_select].copy()

## compile frame-to-frame correlation for each (preselected) cube
def correlation(sci_path, ref_cube_paths, target_table, frame_vect, cube_index, r_in, r_out, 
                sci_select=None):
    if sci_select is None:
        isci = target_table.loc_indices[sci_path]
        sci_select = frame_vect.loc[isci]
        
    sci_frames, mask = format_cube(sci_path, r_in, r_out, select_frames=sci_select)
    sci_id, sci_nframe0 = target_table.loc[sci_path]['Cube_ID','Cube_Nframes']
    corr_mat = []
    corr_index = []
    for refi in cube_index:
        ref_path = ref_cube_paths[refi]
        ref_id, ref_nframe0 = target_table.loc[ref_path]['Cube_ID','Cube_Nframes']
        
        corr_index.append(np.full(ref_nframe0, refi, dtype=int))
        corr_mat.append(get_pcc(sci_frames, ref_path, sci_id, ref_id, r_in, r_out, mask, 
                                sci_nframe0, ref_nframe0, sci_select, frame_vect.loc[refi]))
    return np.concatenate(corr_mat, axis=-1), np.concatenate(corr_index)


## print correlation statistics of reference cube
def print_ref_correlation(ref_corr):
    ## 0:median, 1:min, 2:max, 3:lower quartile, 4:upper quartile
    names = ['median','min','max','lower quartile','upper quartile']
    stats = np.concatenate((np.stack((np.median(ref_corr), np.min(ref_corr), np.max(ref_corr))),
                            np.quantile(ref_corr,(0.25,0.75))))
    stats_str = ', '.join(['{0:s} = {1:.3f}'.format(x,y) for x,y in zip(names,stats)])
    print('> Frame correlation:', stats_str)
