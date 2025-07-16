#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 10:36:07 2025

@author: sstas
"""
import os
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


def stack_cubes(cube_paths, frame_paths):
    
    stacked_cubes = []
    kept_frames = []
    kept_paths = []


    for cube_path, frame_path in zip(cube_paths, frame_paths) : 
        
        #reads the cube
        cube, header = fits.getdata(cube_path, header = True)
        
        #verify dimensions
        n_frames_cube = cube.shape[0]
        
        
        if frame_path == 'na' : 
            selection_vector = np.ones(n_frames_cube, dtype=bool)
        else :
            
           
            selection_vector = fits.getdata(frame_path)
            if selection_vector.ndim > 1:
                selection_vector = selection_vector[:,0]
            
            selection_vector = selection_vector.astype(bool)
            
        n_frames_vector = selection_vector.shape[0]
        
        if n_frames_vector != n_frames_cube:
            raise ValueError(
                f"dimensions mismatch : cube {cube_path} has {n_frames_cube} frames and selection vector {frame_path} has {n_frames_vector} frames")
        
        
        if np.sum(selection_vector) == 0:
            print(f"[Warning] All selection vector values are 0 for {cube_path} skipping.")
            continue
            
        
        
        n_keep = int(np.sum(selection_vector))
        kept_frames.append(n_keep)
            
        
        if n_keep == 0:
            print(f"All selection vector = 0 for cube {cube_path}, skipping this cube")
            continue
        
        #apply selection vector
        selected_cube = cube[selection_vector.astype(bool)]
        
        #stack cube on axis
        stacked_frame = np.nanmean(selected_cube, axis=0)
        
        stacked_cubes.append(stacked_frame)
        kept_paths.append(cube_path)
    
    stacked_array = np.array(stacked_cubes)
    
    return stacked_array, kept_paths, kept_frames



def masked_correlation_matrix(stacked_cubes,r_in, r_out, output_fits='correlation_matrix.fits'):
    
    n= len(stacked_cubes)
    size = stacked_cubes[0].shape[0]
    
    matrix = np.zeros((n,n))
    
    mask = create_mask(size, r_in, r_out)
    
    n_pixels = np.sum(mask)
    
    masked_cubes = np.zeros((n, n_pixels))
    
    stacked_cubes = np.nan_to_num(stacked_cubes, nan=0.0, posinf=0.0, neginf=0.0)

    
   # flattened_cubes = stacked_cubes.reshape(n, -1)
    for i in range(n):
        masked_cubes[i] = mask_frame(stacked_cubes[i], mask)
        print(f"Masked cube {i}: mean={np.nanmean(masked_cubes[i])}, std={np.nanstd(masked_cubes[i])}, any_nan={np.isnan(masked_cubes[i]).any()}, any_inf={np.isinf(masked_cubes[i]).any()}")
  
    for i in range(n):
        for j in range(i, n):
            corr = np.corrcoef(masked_cubes[i], masked_cubes[j])[0, 1]
            matrix[i,j] = corr
            matrix[j, i] = corr #symetrical
            
            
    hdu = fits.PrimaryHDU(data=matrix)
    hdu.writeto(output_fits, overwrite = True)
              
    return matrix
    

    
#%%
if __name__ == '__main__': 
    sofname = "input.sof"
    r_in = 15
    r_out = 35
 
    data_names = {'cube': ('GPI_REDUCED_COLLAPSED_MASTER_CUBE', 2),
                  'frame': ('GPI_FRAME_SELECTION_VECTOR', 0)}

    print("..Getting cube information..")
    cube_paths, frame_select_paths, target_data = get_paths(sofname, data_names)
    
    print("..Reading and stacking cubes..")
    stacked_cubes, kept_paths, frames_kept = stack_cubes(cube_paths, frame_select_paths)

    # Réduction de target_data aux seuls cubes conservés
    target_data = target_data.loc[kept_paths]
    target_data['nframe_keep'] = frames_kept

    ncube = len(stacked_cubes)
    print(f"..{ncube} cubes retained after frame selection..")

    # Initialisation header FITS
    hdr = init_header(ncube, kept_paths[-1], r_in, r_out, np.unique(frame_select_paths))

    for i, path in enumerate(kept_paths):
        target_row = target_data.loc[path]
        update_header(hdr, i, target_row)

    print("..Calculating correlation matrix..")
    corr_matrix = masked_correlation_matrix(stacked_cubes, r_in, r_out, output_fits=None)

    os.makedirs("output", exist_ok=True)
    print("..Saving correlation matrix to FITS..")
    hdu = fits.PrimaryHDU(data=corr_matrix, header=hdr)
    hdu.writeto("output/GPI_CORR_MATRIX-mean_cube_pcc_matrix.fits", overwrite=True)

    print("..Saving target info to CSV..")
    target_data.to_csv("output/GPI_REFERENCE_TARGET_DATA-target_data.csv")
