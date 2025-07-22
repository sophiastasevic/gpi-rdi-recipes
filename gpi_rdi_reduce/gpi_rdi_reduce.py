#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 17:24:48 2024

@author: sstas

RDI reduction of GPI data, with option outputs of azimuthally averaged
contrast and contrast maps.
"""
import argparse
import cv2
import warnings, os, re
import numpy as np
from skimage.draw import disk
from astropy.io import fits
from astropy import units as u
from astropy.utils.exceptions import AstropyWarning

from cube_contrast import get_psf, contrast_df, calc_contrast, calc_contrast_map

warnings.simplefilter('ignore', category=AstropyWarning)
warnings.simplefilter('ignore', category=RuntimeWarning)

D_TEL = 8.2 #m

VALID_NORMS = ['spat-mean','temp-mean','spat-standard','temp-standard','none']

round_up = lambda x: int(x) + (x-int(x)>0)
"""
---------------------------------- FUNCTIONS ----------------------------------
"""
## finds paths in the sof file matching the frame type
def get_path(fnames, ftypes, frame, nmin):
    path = fnames[np.where(ftypes==frame)[0]]
    if len(path) < nmin:
        raise Exception('[Error] Input must contain at least', nmin, frame)
    
    if len(path) > 0:
        return path[0]
    else:
        return 'na'
    
## read in sof file containing file paths and data types and check correct number of the
## required data were input (raises error if not)
def read_sof(sof, data_names):
    data = np.loadtxt(sof,dtype=str)
    fnames = data[:,0] #file names
    ftypes = data[:,1] #file data types

    paths = {}
    for var in data_names.keys():
        paths[var] = get_path(fnames, ftypes, *data_names[var])
    
    return paths

## Read in and fix astropy header (only needed if header is being used for a saved fits file).
def fix_header(path):
    hdul = fits.open(path)
    hdul.verify('fix')
    
    return hdul[0].header


## get indices of data cube frames to keep from frame selection vector if file exists,
## otherwise all frames are used    
def selection_vector(frame_path, nframes_0):
    if frame_path != 'na': 
        select_frames = fits.getdata(frame_path)
        ## sometimes there are two columns, 0: harsher selection, 1: soft selection
        if len(select_frames.shape)>1: 
            select_frames = select_frames[:,0]
        
        select_frames = np.where(select_frames==1.)[0]
   
    else: ## no frame selection vector
        select_frames = np.arange(0,nframes_0)
        
    return select_frames
    

## trims frames + removes bad frames using the frame selection vector
def trim_cube(cube, r_crop, frame_select=None):
    crop = 2 * r_crop
    x,y = cube.shape[-2:]

    if frame_select is not None:
        cube = cube[frame_select]
        
    new_x, new_y = [t if t!=None else (x,y)[i] for i,t in enumerate((crop,crop))]
    if x > new_x:
        ##preserves center of rotation of original (even) cube if new size is odd
        if (x + new_x)%2: 
            x += 1
        cube = cube[..., (x-new_x)//2:(x+new_x)//2, :]
    
    if y > new_y:
        if (y+new_y)%2: 
            y += 1
        cube = cube[..., (y-new_y)//2:(y+new_y)//2]

    return cube

## convert 3D cube to 2D array of size [nframe x npixel] and apply mask  
def format_frames(cube, mask=None):
    nframes, x, y = cube.shape
    data = np.reshape(cube.copy(), (nframes, x*y))
    
    if mask is not None:
        return data[:,mask] 
    else:    
        return data
    
## boolean mask for reshaped data cube
def create_mask(size, inner_radius, outer_radius=None):
    cxy = (size//2, size//2)
    
    if outer_radius is not None:
        mask = np.full((size,size), False)
        mask_out = disk(cxy, outer_radius, shape=(size,size))
        mask[mask_out] = True
    else:
        mask = np.full((size,size), True)
        
    mask_in = disk(cxy, inner_radius, shape=(size,size))
    mask[mask_in] = False

    return mask.reshape((size*size))


## apply centering and/or scaling about the spatial or temporal axis to the data
def normalise(data, method, axis, return_norm=False):
    get_mean = (method!='none'); get_stdev = (method=='standard')

    return apply_norm(data, axis, get_mean, get_stdev, return_norm)

def apply_norm(data, axis, get_mean, get_stdev, return_norm, scale_ref=True):
    ## scale reference observations before temporal normalisation
    if scale_ref == True and axis == 0:
        data/= np.nanstd(data, axis=1).reshape((data.shape[0], 1))
    
    mean = np.nanmean(data, axis=axis)
    stdev = np.nanstd(data, axis=axis)
    zeros = np.where(stdev==0.0)[0]
    stdev[zeros] = 1.0 ##avoid dividing by 0 --> =1 to not scale constant features

    data_ax = np.moveaxis(data, axis, 0)

    if get_mean == True:
        data_ax -= mean

    if get_stdev == True:
        data_ax /= stdev

    if return_norm == True:
        return data, mean, stdev
    else:
        return data

"""
---------------------------------- PCA ----------------------------------
"""
## eigendecomposition of data to get kl vectors
def kl_transform(data):
    ##gram matrix of A.A^T rather A^T.A as latter is too large
    gram = np.dot(data, data.T) 
    nframes = data.shape[0]
    
    ##column v[:,i] is the eigenvector corresponding to the eigenvalue w[i]
    eigen_val, eigen_vect = np.linalg.eig(gram) 
    sort = np.argsort(-eigen_val)
    
    ## if v is the eigenvector of A.A^T, A^T.v is the eigenvector of A^T.A (kernel trick)
    ## KL modes will be a factor of sqrt(eigenval) larger than if calculated without 
    ## the kernel trick so need to normalise as well
    kl_vect = np.dot(eigen_vect.T, data) / np.sqrt(eigen_val).reshape((nframes,1))
    
    return kl_vect[sort]

## carries out PCA reduction on data for each no. PCs specified
def subtract_pcs(science_data, x, y, nframes, ref_data, axis, method, pcs, mean, stdev, cxy_mask):
    ref_data = normalise(ref_data, method, axis)
    
    kl_vect = kl_transform(ref_data)
    kl_projection = np.dot(science_data,kl_vect.T)
    
    pca_cube = np.full((len(pcs), nframes, x, y), np.nan)    
    for i, pc in enumerate(pcs):
        #print("Processing PC:",pc)
        psf_recon = np.dot(kl_projection[:,:pc], kl_vect[:pc]) #np.sum(psf_recon[:pc], axis=0)
        recon_sub = science_data - psf_recon
        
        ##undo normalisation if division by standard deviation
        recon_ax = np.moveaxis(recon_sub,axis,0)
        if method == 'standard':
            recon_ax *= stdev
        
        final = np.full((nframes, x*y), np.nan)
        final[:,cxy_mask] = recon_sub
        
        pca_cube[i] = np.reshape(final, (nframes, x, y))
        
    return pca_cube#, kl_vect[:max(pcs)]

def stack_cube(cube, parang, x, y):
    cxy = (x//2,y//2)

    cube_derot = np.zeros((len(parang), x, y))
    for i in range(0,len(parang)):
        rot = cv2.getRotationMatrix2D(cxy,parang[i],1)
        cube_derot[i] = cv2.warpAffine(cube[i], rot, (x,y), flags=cv2.INTER_LANCZOS4, \
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    reduced_cube = np.nanmedian(cube_derot, axis=0)

    return reduced_cube

## normalises science cube; frame reduction loop
def pca_reduction(sci_frames, ref_frames, ref_select, norm, pcs, mask, size):
    axes = {'spat':1, 'temp':0}
    if norm != 'none':
        axis, method = norm.split('-')
        axis = axes[axis]
    elif norm =='none':
        method = 'none'
        axis = 0

    nframes = sci_frames.shape[0]
    npcs = len(pcs)
    
    ## need to normalise science cube here otherwise if temporal normalisation and single frame reduction,
    ## normalisation will just create a blank frame
    sci_norm, mean, stdev = normalise(sci_frames.copy(), method, axis, return_norm=True)
    
    if ref_frame_select is not None: ## per frame reduction
        pca_cube = np.zeros((npcs, nframes, size, size))
        for i in range(nframes):
            
            ref_select_sci = np.where(ref_select[i]==1)[0]
            
            ## if spatial norm, single value for entire frame; if temporal norm, no. values = x*y
            if 'spat' in norm: 
                frame_mean = mean[i]; frame_std = stdev[i]
            else: 
                frame_mean = mean; frame_std = stdev
                    
            pca_cube_tmp = subtract_pcs(sci_norm[i:i+1], size, size, 1, ref_frames[ref_select_sci].copy(), \
                                        axis, method, pcs, frame_mean, frame_std, mask)
            pca_cube[:,i] = pca_cube_tmp[:,0]
    else: ## reduce full cube at once
        pca_cube = subtract_pcs(sci_norm, size, size, nframes, ref_frames, axis, method, pcs, mean, stdev, mask)
        
    return pca_cube


## crops reference cube and recrops science cube if necessary and initiates pca reduction
def pca(cube, ref_cube, norm, pcs, r_crop, ref_frame_select, r_mask, nref):
    ref_size = ref_cube.shape[-1]
    if ref_size < r_crop*2: 
        r_crop = ref_size//2
        cube = trim_cube(cube, r_crop)
        print('[Warning] Reference cube size smaller than input crop radius, changing crop radius to {0:d} pixels'.format(r_crop))
    else:
        ref_cube = trim_cube(ref_cube, r_crop)
    
    mask = create_mask(r_crop*2, inner_radius=r_mask, outer_radius=r_crop)
    
    data_stack = np.sum(np.concatenate((ref_cube,cube), axis=0), axis=0).flatten()
    mask_nan = np.where(~np.isfinite(data_stack))[0]
    
    mask[mask_nan] = False
    
    sci_frames = format_frames(cube, mask)
    ref_frames = format_frames(ref_cube, mask)
        
    if (ref_frame_select==1).all():
        ## can reduce the full cube at once
        ref_frame_select = None

    return pca_reduction(sci_frames, ref_frames, ref_frame_select, norm, pcs, mask, r_crop*2)

def format_array(in_list):
    if in_list is None:
        return None
    elif '-' in in_list:
        if ',' not in in_list:
            step = 1
        else:
            step = int(in_list.split(',')[-1])
        a_range = [int(x) for x in re.findall(r'[0-9]+', in_list.split(',')[0])]
        if len(a_range) == 1:
            array = np.arange(stop=a_range[0]+step, step=step)
        else:
            array = np.arange(stop=a_range[1]+step, start=a_range[0], step=step)
        ## avoid PC = 0
        if array[0] == 0:
            if step == 1:
                array = array[1:]
            else:
                array[0] = 1 
        array[-1] = a_range[-1]
    else:
        array = np.array([int(x) for x in re.findall(r'[0-9]+', in_list)])
    return array

def update_pcs(pcs, min_nref):
    if pcs is None:
        pcs_tmp = np.arange(1, min_nref+1)
    elif max(pcs) > min_nref:
        pcs_tmp = pcs[pcs<=min_nref]
    else:
        pcs_tmp = pcs

    if len(pcs_tmp) < 10:
        pc_str = str(pcs_tmp)
    else:
        pc_dif = np.diff(pcs_tmp)
        if np.all(pc_dif==pc_dif[0]):
            pc_str = '{0:d}-{1:d} [{2:d}]'.format(pcs_tmp[0],pcs_tmp[-1],pc_dif[0])
        else:
            pc_str = str(pcs_tmp[0])
            for i in range(1, len(pcs_tmp)-1):
                pc_dif = np.diff(pcs_tmp[i-1:i+2])
                if pc_dif[0] == pc_dif[1]:
                    if pc_str[-1]!='-': 
                        pc_str+='-'
                    step = pc_dif[0]
                elif pc_str[-1]=='-':
                    pc_str+='{0:d}'.format(pcs_tmp[i])
                else:
                    pc_str+=',{0:d}'.format(pcs_tmp[i])
            if pc_dif[-1] != step: 
                pc_str+=','        
            pc_str+='{0:d} [{1:d}]'.format(pcs_tmp[-1], step)
            
        pc_str = (pc_str, "[step size between PCs]")
    
    return pcs_tmp, pc_str
#%%
"""
---------------------------------- MAIN CODE ----------------------------------
"""
if __name__=='__main__':
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('sof', help='name of sof file', type=str)
    parser.add_argument('--pca_norm', help='Normalisation method for PCA reduction; options: spat-mean, temp-mean, spat-standard, temp-standard, none', type=str, default='none')
    parser.add_argument('--pc_list', help='Number of PCs to subtract. For multiple, either comma seperated list or dash seperated (incl.) range (opt. comma seperated step size, e.g. 50-100,10 = [50,60,70,80,90,100]; pc0 automatically changed to pc1).',  default='50-250,50')
    parser.add_argument('--r_crop', help='Crop frame to this pixel radius.', type=int)
    parser.add_argument("--contrast", action='store_true', help='Calculate radial contrast of the reduced cube.')
    parser.add_argument("--contrast_map", action='store_true', help='Create contrast map of reduced cube.')
    parser.add_argument('--save_residuals', action='store_true', help='Output residuals cube (uncombined PSF subtracted science frames) as a temporary file.')
    
    args = parser.parse_args()
    """
    from input_class import Input_GPI_RDI
    args = Input_GPI_RDI('input.sof', 'spat-mean', '10-40,10', 100)
    #%%    
    ## file containing input data paths and types
    sof = args.sof
    
    ## normalisation method to be applied to data cube before pca reduction
    norm = (args.pca_norm).strip()
    if norm not in VALID_NORMS:
        raise ValueError("'{0:s}' is not a recognised normalisation method, please choose from: '{1:s}'" \
                         .format(norm,"', '".join(VALID_NORMS)))
    
    ## list of principal components to use in reduction
    pc_list = format_array(args.pc_list)
    
    ## new frame size
    r_crop = args.r_crop
     
    ## output residuals cube
    save_residuals = args.save_residuals
    
    ## calculate contrast
    cont_map = args.contrast_map
    cont_csv = args.contrast
    
    data = np.loadtxt(sof,dtype=str)
    fnames = data[:,0]
    ftypes = data[:,1]
    
    ## ----- Change these data types to the GPI equivalent -----
    data_names = {'science':('GPI_REDUCED_COLLAPSED_MASTER_CUBE',1),
                  'parang':('GPI_SCIENCE_PARA_ROTATION_CUBE',1),
                  'psf':('GPI_SCIENCE_PSF_MASTER_CUBE',0+(cont_map|cont_csv)),
                  'lambda':('GPI_SCIENCE_LAMBDA_INFO',0+(cont_map|cont_csv)),
                  'frame_select':('GPI_FRAME_SELECTION_VECTOR',0),
                  'ref_lib':('GPI_REFERENCE_CUBE',1),
                  'ref_select':('GPI_REFERENCE_FRAME_SELECTION_VECTOR',1)}
    
    paths = read_sof(sof, data_names)
    nref = len(paths['ref_lib'])
    
    print('.. Reading in science data ..')
    
    sci_nframe0 = fits.getheader(paths['science'])['NAXIS3']
    select_frames = selection_vector(paths['frame_select'], sci_nframe0)
    sci_nframe = len(select_frames)
    
    parang = fits.getdata(paths['parang'])[select_frames]
    
    science_cube = trim_cube(fits.getdata(paths['science']), r_crop, select_frames)
    pxscale = fits.getheader(paths['science'])['PIXTOARC']
    r_mask = round_up(123.35/pxscale) 
    
    if (cont_map|cont_csv):
        wl_lambda = fits.getdata(paths['lambda'])
        wl_to_radius = lambda wl, x: u.rad.to(u.mas,x*wl*1e-6/D_TEL)/pxscale
        
        d_aper = np.mean([wl_to_radius(wl,1) for wl in wl_lambda])
        
        ## measure psf fwhm
        fwhm, psf, psf_max = get_psf(paths['psf'], d_aper)
        df_cont = None
    #%%
    ref_cube, hdr = fits.getdata(paths['ref_lib'], header=True)    
    nref = hdr['NCORR']
        
    ref_frame_select = fits.getdata(paths['ref_select'])
    ## check number of science frames in the reference frame selection vector is correct
    ref_sci_nframe = ref_frame_select.shape[0]
    if ref_sci_nframe == sci_nframe0 and sci_nframe0 > sci_nframe:
        ref_frame_select = ref_frame_select[select_frames]
    
    elif ref_sci_nframe == sci_nframe:
        next
        
    else:
        raise Exception("[Error] Mismatch between number of science frames in cube (%d) and \
                          reference frame selection vector (%d)" % (sci_nframe, ref_sci_nframe))

    print(".. Running {0:s} PCA reduction".format(norm))
    pcs, pc_str = update_pcs(pc_list, nref)
    
    hdr['PC_LIST'] = pc_str
    hdr['PCA_NORM'] = norm
    hdr['PCA_MASK'] = r_mask
    if (cont_map|cont_csv):
        hdr["FWHM"] = (fwhm, "mean FWHM of stellar PSF [px]")
        hdr["D_APER"] = (d_aper, "mean diameter of aperture used to sum PSF [lambda/D in px]")
        hdr["SUM_PSF"] = (psf, "mean sum of stellar PSF within a central aperture of diameter D_APER [ADU/s]")
        hdr["MAX_PSF"] = (psf_max, "mean maximum stellar PSF value [ADU/s]")
            
    pca_cube = pca(science_cube, ref_cube, norm, pcs, r_crop, ref_frame_select, r_mask, nref)
    
    print("> Derotating and stacking frames.")
    npcs, nframes, x, y = pca_cube.shape
    reduced_cube = np.zeros((npcs, x, y))
    for ipc in range(npcs):
        reduced_cube[ipc] = stack_cube(pca_cube[ipc], parang, x, y)
    
    print('> Saving reduced cube.')
    hdu = fits.PrimaryHDU(data=reduced_cube, header=hdr)
    #%%
    hdu.writeto('reduced_cube_stack.fits', overwrite=True) 
    
    if save_residuals:
        print("> Saving residuals cube.")
        hdu_full = fits.PrimaryHDU(data=pca_cube, header=hdr)
        hdu_full.writeto('residuals.fits', overwrite=True) 
            
    if cont_map:
        print('..Calculating contrast map..')
        contrast_map = calc_contrast_map(reduced_cube, x, fwhm, psf_max)
        
        print('> Saving contrast map.')
        hdu_cm = fits.PrimaryHDU(data=contrast_map, header=hdr)
        hdu_cm.writeto('5-sig_contrast_map_stack.fits', overwrite=True) 

    if cont_csv:
        print('..Calculating contrast..')
        contrast, dist = calc_contrast(reduced_cube, d_aper, x, psf, r_mask=r_mask)
        df_cont = contrast_df(contrast, dist, pcs, d_aper)
        
        print('> Saving contrast data.')
        df_cont.to_csv('5-sig_contrast_stack.csv')
        
            
    
        