#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 10:47:59 2025

@author: sstas
"""
import warnings
import numpy as np
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning

import pandas as pd
import photutils  
from photutils.aperture import CircularAperture, CircularAnnulus

from vip_hci.var import fit_2dgaussian

warnings.simplefilter('ignore', category=AstropyWarning)
warnings.simplefilter('ignore', category=RuntimeWarning)

round_up = lambda x: int(x) + (x-int(x)>0)

## calculate the background flux of the stellar psf frame
def background_flux(psf, cx, cy):
    annul_bg = CircularAnnulus((cx,cy), cx-2, cx)
    bg_flux = photutils.aperture_photometry(psf, annul_bg, method = 'exact')
    
    return bg_flux['aperture_sum']/annul_bg.area
    
## correct psf background and measure fwhm, sum within an aperture, and maximum
def calc_fwhm(psf, x, y, d_aper):
    ## makes sure the psf background has a mean of 0
    psf -= background_flux(psf, x//2, y//2)
    DF_fit = fit_2dgaussian(psf, crop=False, debug=False)
    fwhm = np.mean([DF_fit['fwhm_x'],DF_fit['fwhm_y']])
        
    aper = CircularAperture((x//2,y//2), d_aper/2)
    psf_sum = photutils.aperture_photometry(psf, aper)["aperture_sum"]
    psf_max = np.nanmax(psf)
    
    return fwhm, psf_sum, psf_max

## read in psf + return fwhm, sum within an aperture, and maximum
def get_psf(psf_path, d_aper):
    psf, psf_hdr = fits.getdata(psf_path, header=True)
    if 'WARNING' in psf_hdr or 'WARNING1' in psf_hdr:
        raise Exception("[Error] Input PSF is a dummy PSF.")
    
    if psf.ndim > 3: ## psf image before and after ADI sequence
        psf = np.nanmean(psf, axis=(0,1))
    else:
        psf = np.nanmean(psf, axis=0)
    
    x, y = psf.shape
    
    fwhm, psf_sum, psf_max = calc_fwhm(psf, x, y, d_aper)
    psf_sum = psf_sum[0]
    print("Channel mean: lambda/D = {0:.2f} px, sum of PSF in aperture = {1:.2f} ADU".format(d_aper, psf_sum))
    
    return fwhm, psf_sum, psf_max

## generate non-overlapping circular apertures that span the circumference of a circle with
## a specified radius, centered at the middle of the frame
def generate_apertures(r, d_aper, size):
    ## find center xy position of each aperture that fits in an annulus of radius r
    n_aper = (2*np.pi*r)/d_aper
    ang_sep = 360/n_aper
    x = np.zeros(int(n_aper)); y = np.zeros_like(x)
    for i in range(int(n_aper)):
        x[i] = r * np.cos(np.deg2rad(i*ang_sep)) + size//2
        y[i] = r * np.sin(np.deg2rad(i*ang_sep)) + size//2
    
    return CircularAperture([(x_tmp,y_tmp) for x_tmp,y_tmp in zip(x,y)], d_aper/2)

## measure the standard deviation of the sum of fluxes within apertures
def calc_contrast(data, d_aper, size, psf, sigma=5, r_mask=0, dist=None):
    if dist is None:
        dist = np.arange(r_mask + d_aper/2, (size-d_aper)/2, d_aper)
    npcs = data.shape[0]; ndist = len(dist)
    
    contrast = np.zeros((ndist,npcs))
    for i,r in enumerate(dist):
        apers = generate_apertures(r, d_aper, size)
        for ipc in range(npcs):
            fluxes = photutils.aperture_photometry(data[ipc], apers)
            fluxes = np.array(fluxes["aperture_sum"])
            contrast[i,ipc] = np.nanstd(fluxes)
    
    return sigma*contrast/psf, dist

def contrast_df(contrast, dist, pcs, d_aper):
    index = pd.MultiIndex.from_product([dist,[d_aper]], names=['separation[px]','aperture_diam'])
    df = pd.DataFrame(contrast, index=index, columns=pcs)
    
    return df.rename_axis(columns="KL_modes")

## create a contrast map using the standard deviation of a fwhm sized square aperture
## around each pixel in the frame
def calc_contrast_map(data, size, fwhm, psf_mean, sigma=5):
    contrast_map = np.zeros_like(data) 
    xwin = ywin = int(fwhm)
    for x in range(size):
        xmin = max([0, x - xwin])
        xmax = min([size, x + xwin+1]) ## +1 to account for non-inclusive endpoint indexing
        for y in range(size):
            ymin = max([0, y - ywin])
            ymax = min([size, y + ywin+1])

            contrast_map[...,x,y] = np.nanstd(data[...,xmin:xmax,ymin:ymax], axis=(-2,-1))
    
    return sigma*contrast_map/psf_mean
    
    