#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 17:24:48 2024

@author: sstas

RDI reduction of SPHERE-IRDIS data, with option outputs of azimuthally averaged
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
    elif len(path) == 1 and 'REFERENCE' not in frame:
        return path[0]
    else:
        return list(path)

## read in sof file containing file paths and data types and check correct number of the
## required data were input (raises error if not)
def read_sof(sof, data_names):
    data = np.loadtxt(sof,dtype=str)
    fnames = data[:,0] #file names
    ftypes = data[:,1] #file data types

    paths = {}
    for var in data_names.keys():
        paths[var] = get_path(fnames, ftypes, *data_names[var])
    
    ##star hopping ref lib
    if len(paths['science']) == 2:
        sci_obj = fits.getheader(paths['parang'])['OBJECT']
        for sci_var, ref_var in zip(['science','frame_select'],['ref_lib','ref_select']):
            for i,f in enumerate(paths[sci_var]):
                if fits.getheader(f)['OBJECT'] != sci_obj: 
                    paths[ref_var].append(paths[sci_var].pop(i)+'_STAR-HOP')
            paths[sci_var] = paths[sci_var][0]
        
    elif len(paths['ref_lib']) == 0:
        raise Exception('[Error] Input must contain at least 1 reference library')
        
    elif type(paths['science']) == list and len(paths['science']) > 2:
        raise Exception('[Error] Can only handle 1 reference library input as IRD_SCIENCE_REDUCED_MASTER_CUBE')
        
    return paths

## Read in and fix astropy header (only needed if header is being used for a saved fits file).
def fix_header(path):
    hdul = fits.open(path)
    hdul.verify('fix')
    
    return hdul[0].header

## if star-hopping, copy over ref star info to science header + other headers that would be
## in a non star-hopping reference library but don't appear in the science header
def star_hopping_header(sci_path, ref_hdr, nframes_ref):
    hdr = fix_header(sci_path)
    hdr['PARAM'] = 'STAR-HOPPING'
    hdr.set('NCUBE', 1, 'total number of ref targets used')
    hdr.set('', '----------------', after='NCUBE')
    
    hdr['OBJ_REF'] = ref_hdr['OBJECT']
    hdr['OBS_REF'] = ref_hdr['DATE-OBS']
    hdr['ID_REF'] = (ref_hdr['DATASUM'], 'Unique identifier taken from cube DATASUM header')
    hdr['N_REF'] = (nframes_ref, 'Number of frames used from this cube')
    
    return hdr

## get indices of data cube frames to keep from frame selection vector if file exists,
## otherwise all frames are used    
def selection_vector(frame_path, nframes):
    try:
        frame_select = fits.getdata(frame_path)
        
    except FileNotFoundError:
        frame_select = np.ones(nframes)
        print('>> Warning: Could not find frame selection vector file {0:s}'.format(paths['frame_select']))
    
    if len(frame_select.shape)>1:
        frame_select=frame_select[:,0]
    
    return frame_select

## trims frames + removes bad frames using the frame selection vector
def trim_cube(cube, crop, frame_select=None):
    x,y = cube.shape[-2:]

    if frame_select is not None:
        cube = cube[:,frame_select]
        
    new_x, new_y = [t if t!=None else (x,y)[i] for i,t in enumerate((crop,crop))]
    if x > new_x:
        ##preserves center of rotation of original (even) cube if new size is odd
        if (x+new_x)%2: 
            x+=1
        cube = cube[...,(x-new_x)//2:(x+new_x)//2,:]
    
    if y > new_y:
        if (y+new_y)%2: 
            y+=1
        cube = cube[...,(y-new_y)//2:(y+new_y)//2]

    return cube

## convert 3D cube to 2D array of size [nframe x npixel] and apply mask  
def create_sci(cube, r_mask=None):
    nframes, x, y = cube.shape
    data = np.reshape(cube.copy(),(nframes,x*y))
    
    if r_mask is not None:
        cxy_mask = create_mask(x, r_mask)
        return data, cxy_mask    
    else:    
        return data
    
## boolean mask for reshaped data cube
def create_mask(size, inner_radius, outer_radius=None):
    cxy=(size//2,size//2)
    
    if outer_radius is not None:
        mask=np.full((size,size),False)
        mask_out=disk(cxy,outer_radius,shape=(size,size))
        mask[mask_out]=True
    else:
        mask=np.full((size,size),True)
        
    mask_in=disk(cxy,inner_radius,shape=(size,size))
    mask[mask_in]=False

    return mask.reshape((size*size))


## apply centering and/or scaling about the spatial or temporal axis to the data
def normalise(data, method, axis, return_norm=False):
    get_mean = (method!='none'); get_stdev = (method=='standard')

    return apply_norm(data, axis, get_mean, get_stdev, return_norm)

def apply_norm(data, axis, get_mean, get_stdev, return_norm, scale_ref=True):
    ## scale reference observations before temporal normalisation
    if scale_ref == True and axis == 0:
        data/= np.nanstd(data,axis=1).reshape((data.shape[0],1))
    
    mean = np.nanmean(data,axis=axis)
    stdev = np.nanstd(data,axis=axis)
    zeros = np.where(stdev==0.0)[0]
    stdev[zeros] = 1.0 ##avoid dividing by 0 --> =1 to not scale constant features

    data_ax = np.moveaxis(data,axis,0)

    if get_mean == True:
        data_ax-= mean

    if get_stdev == True:
        data_ax/= stdev

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
    gram = np.dot(data,data.T) 
    nframes = data.shape[0]
    
    ##column v[:,i] is the eigenvector corresponding to the eigenvalue w[i]
    eigen_val, eigen_vect = np.linalg.eig(gram) 
    sort = np.argsort(-eigen_val)
    
    ## if v is the eigenvector of A.A^T, A^T.v is the eigenvector of A^T.A (kernel trick)
    ## KL modes will be a factor of sqrt(eigenval) larger than if calculated without 
    ## the kernel trick so need to normalise as well
    kl_vect = np.dot(eigen_vect.T,data)/np.sqrt(eigen_val).reshape((nframes,1))
    
    return kl_vect[sort]

## carries out PCA reduction on data for each no. PCs specified
def subtract_pcs(science_data, x, y, nframes, ref_data, axis, method, pcs, mean, stdev, cxy_mask):
    ref_data = normalise(ref_data, method, axis)
    kl_vect = kl_transform(ref_data)
    kl_projection = np.dot(science_data,kl_vect.T)
    
    pca_cube = np.zeros((len(pcs), nframes, x, y))    
    for i, pc in enumerate(pcs):
        #print("Processing PC:",pc)
        psf_recon = np.dot(kl_projection[:,:pc],kl_vect[:pc]) #np.sum(psf_recon[:pc], axis=0)
        recon_sub = science_data-psf_recon
        
        ##undo normalisation if division by standard deviation
        recon_ax = np.moveaxis(recon_sub,axis,0)
        if method == 'standard':
            recon_ax*= stdev
        
        final = np.zeros((nframes, x*y))
        final[:,cxy_mask] = recon_sub
        
        pca_cube[i] = np.reshape(final, (nframes,x,y))
        
    return pca_cube#, kl_vect[:max(pcs)]

def stack_cube(cube, parang, x, y):
    cxy = (x//2,y//2)

    cube_derot = np.zeros((len(parang),x,y))
    for i in range(0,len(parang)):
        rot = cv2.getRotationMatrix2D(cxy,parang[i],1)
        cube_derot[i] = cv2.warpAffine(cube[i],rot,(x,y),flags=cv2.INTER_LANCZOS4, \
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    reduced_cube = np.nanmedian(cube_derot,axis=0)

    return reduced_cube

## normalises science cube; frame reduction loop
def pca_reduction(data_cube, norm, pcs, r_mask, ref_cube, crop, nref, ref_select=None):
    axes = {'spat':1, 'temp':0}
    if norm != 'none':
        axis, method = norm.split('-')
        axis = axes[axis]
    elif norm =='none':
        method = 'none'
        axis = 0

    nframes, x, y = data_cube.shape
    npcs = len(pcs)
    
    science_cube, cxy_mask = create_sci(data_cube, r_mask)
    ## need to normalise science cube here otherwise if temporal normalisation and single frame reduction,
    ## normalisation will just create a blank frame
    sci_norm, mean, stdev = normalise(science_cube[:,cxy_mask].copy(), method, axis, return_norm=True)
    
    ref_lib = create_sci(ref_cube)
    
    if ref_select is not None: ## per frame reduction
        pca_cube = np.zeros((npcs,nframes,x,y))
        #kl_vects = np.zeros((nframes,max(pcs),cxy_mask.sum()))
        for i in range(nframes):
            #print("Processing frame {0:d}/{1:d}".format(i,nframes))
            ref_frames = np.where(ref_select[i]==1)[0]
            ## if spatial norm, single value for entire frame; if temporal norm, no. values = x*y
            if 'spat' in norm: 
                normi = i
            else: 
                normi = np.arange(sci_norm.shape[1])
                    
            pca_cube_tmp = subtract_pcs(sci_norm[i:i+1], x, y, 1, ref_lib[ref_frames][:,cxy_mask].copy(), \
                                        axis, method, pcs, mean[normi], stdev[normi], cxy_mask)
            pca_cube[:,i] = pca_cube_tmp[:,0]
    else: ## reduce full cube at once
        pca_cube = subtract_pcs(sci_norm, x, y, nframes, ref_lib[:,cxy_mask], axis, method, pcs, mean, stdev, cxy_mask)
        
    return pca_cube#, kl_vects


## crops reference cube and recrops science cube if necessary; wavelength channel reduction loop
def pca(cube, ref_cube, norm, pcs, crop, ref_frame_select, r_mask, nref):
    if ref_cube.shape[-1] < crop: 
        crop = ref_cube.shape[-1]
        cube = trim_cube(cube, crop)
        print('>> Warning: Reference cube size smaller than input crop size, changing crop size to {0:d} pixels'.format(crop))
    else:
        ref_cube = trim_cube(ref_cube, crop)
    
    if ref_frame_select.ndims>1:
        frame_select = ref_frame_select
    else:
        frame_select = None

    return pca_reduction(cube, norm, pcs, r_mask, ref_cube, crop, nref, frame_select)

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
        pcs_tmp = np.append(pcs[np.where(pcs<min_nref)[0]], min_nref)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('sof', help='name of sof file', type=str)
    parser.add_argument('--pca_norm', help='Normalisation method for PCA reduction; options: spat-mean, temp-mean, spat-standard, temp-standard, none', type=str, default='none')
    parser.add_argument('--pc_list', help='Number of PCs to subtract. For multiple, either comma seperated list or dash seperated (incl.) range (opt. comma seperated step size, e.g. 50-100,10 = [50,60,70,80,90,100]; pc0 automatically changed to pc1).',  default='50-250,50')
    parser.add_argument('--crop', help='Crop frame xy to this size.', type=int)
    parser.add_argument("--contrast", action='store_true', help='Calculate radial contrast of the reduced cube.')
    parser.add_argument("--contrast_map", action='store_true', help='Create contrast map of reduced cube.')
    parser.add_argument('--save_residuals', action='store_true', help='Output residuals cube (uncombined PSF subtracted science frames) as a temporary file.')
    
    args = parser.parse_args()
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
    crop = args.crop
     
    ## output residuals cube
    save_residuals = args.save_residuals
    
    ## calculate contrast
    cont_map = args.contrast_map
    cont_csv = args.contrast
    
    data = np.loadtxt(sof,dtype=str)
    fnames = data[:,0]
    ftypes = data[:,1]
    
    ## ----- Change these data types to the GPI equivalent -----
    data_names = {'science':('IRD_SCIENCE_REDUCED_MASTER_CUBE',1),
                  'parang':('IRD_SCIENCE_PARA_ROTATION_CUBE',1),
                  'psf':('IRD_SCIENCE_PSF_MASTER_CUBE',0+(cont_map|cont_csv)),
                  'lambda':('IRD_SCIENCE_LAMBDA_INFO',0+(cont_map|cont_csv)),
                  'frame_select':('IRD_FRAME_SELECTION_VECTOR',0),
                  'ref_lib':('IRD_REFERENCE_CUBE',0),
                  'ref_select':('IRD_REFERENCE_FRAME_SELECTION_VECTOR',0)}
    
    paths = read_sof(sof, data_names)
    nref = len(paths['ref_lib'])
    
    print('.. Reading in science data ..')
    
    ## ----- Check that NAXIS3 is the number of frames in the cube -----
    sci_nframes = fits.getheader(paths['science'])['NAXIS3']
    frame_select = selection_vector(paths['frame_select'], sci_nframes)
    good_frames = np.where(frame_select==1.)[0]
    
    parang = fits.getdata(paths['parang'])[good_frames]
    
    science_cube = trim_cube(fits.getdata(paths['science']), crop, good_frames)
    pxscale = fits.getheader(paths['science'])['PIXTOARC']
    r_mask = round_up(92/pxscale) 
    
    if (cont_map|cont_csv):
        wl_lambda = fits.getdata(paths['lambda'])
        wl_to_radius = lambda wl, x: u.rad.to(u.mas,x*wl*1e-6/D_TEL)/pxscale
        
        d_aper = np.mean([wl_to_radius(wl,1) for wl in wl_lambda])
        
        ## measure psf fwhm
        fwhm, psf, psf_max = get_psf(paths['psf'], d_aper)
        df_cont = None
    #%%
    for ref_path in paths['ref_lib']:
        ref_cube, hdr = fits.getdata(ref_path.replace('_STAR-HOP',''), header=True)
        
        if 'STAR-HOP' in ref_path:
            param = 'STAR-HOPPING'
            try:
                ref_select_path = [x[:-9] for x in paths['ref_select'] if 'STAR-HOP' in x][0]
                ref_frame_select = np.expand_dims(selection_vector(ref_select_path, ref_cube.shape[1]), axis=0)
            except IndexError:
                print('[Warning] No frame selection vector input for star-hopping library.')
                ref_frame_select = np.ones(ref_cube.shape[0])
            
            nref = len(ref_frame_select)
            hdr = star_hopping_header(paths['science'], hdr, nref)
        else: ## standard reference library
            param = hdr['PARAM']
            if param == 'RAND':
                nref = hdr['NREF']
            else:    
                nref = hdr['NCORR']
            
            try:
                ref_select_path = [x for x in paths['ref_select'] if param+'.' in x and
                                   os.path.dirname(x) == os.path.dirname(ref_path)][0]
            except IndexError:
                print('[Warning] Reference frame selection vector not in input, skipping {0:s} reference library.'.format(param))
                continue
                
            ref_frame_select = fits.getdata(ref_select_path)
            if ref_frame_select.ndims > 1:
                if ref_frame_select.shape[0] != len(good_frames):
                    ref_frame_select = ref_frame_select[:,good_frames]

        print("> Running {0:s} PCA reduction with {1:s} reference library.".format(norm,param))
        pcs, pc_str = update_pcs(pc_list, np.min(np.sum(ref_frame_select, axis=-1)))
        
        hdr['PC_LIST'] = pc_str
        hdr['PCA_NORM'] = norm
        hdr['PCA_MASK'] = r_mask
        if (cont_map|cont_csv):
            hdr["FWHM"] = (fwhm, "mean FWHM of stellar PSF [px]")
            hdr["D_APER"] = (d_aper, "mean diameter of aperture used to sum PSF [lambda/D in px]")
            hdr["SUM_PSF"] = (psf, "mean sum of stellar PSF within a central aperture of diameter D_APER [ADU/s]")
            hdr["MAX_PSF"] = (psf_max, "mean maximum stellar PSF value [ADU/s]")
                
        path_end = '{0:s}_stack'.format(param)
        pca_cube = pca(science_cube, ref_cube, norm, pcs, crop, ref_frame_select, r_mask, nref)
        
        print("Combining frames.")
        npcs, nframes, x, y = pca_cube.shape
        reduced_cube = np.zeros((npcs,x,y))
        for ipc in range(npcs):
            reduced_cube[ipc] = stack_cube(pca_cube[ipc], parang, x, y)
        
        print('> Saving reduced cube.')
        hdu = fits.PrimaryHDU(data=reduced_cube, header=hdr)
        hdu.writeto('reduced_cube_{0:s}.fits'.format(path_end)) 
        
        if save_residuals:
            print("> Saving residuals cube.")
            hdu_full = fits.PrimaryHDU(data=pca_cube, header=hdr)
            hdu_full.writeto('residuals_{0:s}.fits'.format(path_end)) 
                
        if cont_map:
            print('..Calculating contrast map..')
            contrast_map = calc_contrast_map(reduced_cube, x, fwhm, psf_max)
            
            print('> Saving contrast map.')
            hdu_cm = fits.PrimaryHDU(data=contrast_map, header=hdr)
            hdu_cm.writeto('5-sig_contrast_map_{0:s}.fits'.format(path_end)) 
    
        if cont_csv:
            print('..Calculating contrast..')
            contrast, dist = calc_contrast(reduced_cube, d_aper, x, psf, r_mask=r_mask)
            df_cont = contrast_df(contrast, dist, pcs, param, df_prev=df_cont)
            
    if cont_csv:
        if len(paths['ref_lib']) == 1:
            path_end = "_"+path_end
        else:
            path_end = "_stack"
        
        print('> Saving contrast data.')
        df_cont.to_csv('5-sig_contrast{0:s}.csv'.format(path_end))
        
            
    
        