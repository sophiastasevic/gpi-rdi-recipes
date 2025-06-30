#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preselect reference cubes/frames that most closely match the parameter values of the science frame(s).

Finds absolute difference between reference and science cube/frame parameters, and returns cubes/
frames with the smallest difference.
"""
import numpy as np
#from astropy.time import Time

## units of the parameters
P_UNIT = {'SEEING': 'arcsec',
          'TAU0': 'ms',
          'EPOCH': 'yr',
          'LWE': 'm/s',
          'WDH': ['m/s','deg'],
          'ELEV': 'deg',
          'DIT': 's',
          'SPEED': 'm/s',
          'DIR': 'deg'}

## colum names in asm_data csv or target_table for each parameter
param_names = {'SEEING': 'seeing',
               'TAU0': 'tau0',
               'EPOCH': 'Epoch_decimalyear', #'timestamp_mjd',
               'LWE': 'wind_speed_30m', #'wind_dir_30m_onframe'],
               'WDH': ['wind_speed_200mbar','wind_dir_200mbar_onframe'],
               'ELEV': 'elevation',
               'DIT': 'Exptime',
               'NDIT': 'NDIT',
               'MG': 'G_mag',
               'MH': 'H_mag',
               'MK': 'K_mag',
               'SPT': 'SpT_decimal'}

## parameters which may vary by frame, rather than being the same for a full observation
PER_FRAME = ['SEEING','TAU0','LWE','WDH','ELEV']

"""
---------------------------------- FUNCTIONS ----------------------------------
""" 
## preselection for parameter that applies to the full cube, using the target info table.
## [DIT, NDIT, MG, MH, MK, SPT]
def per_target(target_table, isci, param, param_max, ncorr, nframes, best_params=None):
    colname = param_names[param]
    ref_params, sci_param, delta_p = target_values(target_table, isci, colname)
    
    ## discreet values, find all in same group
    if param == 'DIT':
        param_text = '> Science value = {0:.1f}{1:s}'.format(sci_param, target_table[colname].unit)
        ref_select = np.where(ref_params==sci_param)[0]
    elif param == 'NDIT': ## also discreet, but may not have enough frames to meet the ncorr limit
        param_text = '> Science value = {0:d}'.format(int(sci_param))        
        ref_select = ndit(delta_p, ncorr, nframes)
        
    ## continuous values, find set number of best matching
    else:
        param_text = '> Science value = {0:.2f}'.format(sci_param)
        param_sort = np.argsort(delta_p) ## sort by smallest difference, i.e. closest match
        
        if param == 'SPT':
            ## limit spectral type selection to be of the same luminosity class
            ref_tmp, sci_tmp, delta_tmp = target_values(target_table, isci, colname.replace('SpT','Lumi'))
            class_match = np.where(delta_tmp < 1)[0]
            if np.sum(nframes[class_match]) < param_max:
                class_match = np.where(delta_tmp <= 1)[0]
                print("[Warning] Matching luminosity class +/-1 with respect to the science target")
                    
            param_text = '> Science value = {0:.2f} (lum class = {1:.2f})'.format(sci_param, sci_tmp)
            param_sort = np.array([x for x in param_sort if x in class_match])
        
        ## select number of best matching cubes that have at least param_max frames between them
        frm_count = 0
        for i, index in enumerate(param_sort):
            frm_count += nframes[index]
            if frm_count >= param_max:
                ref_select = param_sort[:i+1]
                break    
        best_params = ref_params[ref_select]
    
    ## print science value and number of frames, cubes, and (if applicable) parameter range of preselection
    print('{0:s}; no. preselected frames = {1:d} (from {2:d} targets)'.format(
            param_text, np.sum(nframes[ref_select]), len(ref_select)))
    if best_params is not None:
        print('> Preselected reference values: median = {0:.2f}, max = {1:.2f}, min = {2:.2f} \n'.format(
                np.nanmedian(best_params), np.nanmax(best_params), np.nanmin(best_params)))
    else:
        print('\n')
    
    return ref_select

## get reference and science parameter values from the table and calculate the absolute difference
def target_values(target_table, isci, colname):
    ref_params = np.array(target_table[colname])
    sci_param = ref_params[isci]
    if target_table['Master_ref'][isci] == False: ## remove science row from table if not using as ref
        ref_params = np.delete(ref_params, isci)
    
    return ref_params, sci_param, np.abs(ref_params-sci_param)

## discreet selection of NDIT may not have enough frames to meet the ncorr limit, so increase
## the allowed NDIT difference until it is met
def ndit(delta_p, ncorr, nframes):
    same_ndit = np.where(delta_p == 0)[0]
    if np.sum(nframes[same_ndit]) >= ncorr:
        return same_ndit
    else:
        ## all unique delta_p values in ascending order
        offsets = np.sort(np.array(list(set(delta_p))))
        for o in offsets:
            select = np.where(delta_p<=o)[0]
            if np.sum(nframes[select]) >= ncorr:
                print("[Warning] Setting NDIT offset selection threshold to {0:.1f}.".format(o))
                return select              
            
"""
--------------------------------------------------------------------------------
""" 
## preselection for parameter that is unique for each frame, using asm + timestamp dataframe.
## [SEEING, TAU0, EPOCH, LWE, WDH]
def per_frame(asm_data, isci, param, param_max, USE_SCI=False):
    colname = param_names[param]

    cube_index = asm_data.index.get_level_values(0)
    sci_data = asm_data.loc[isci,colname].to_numpy()
    if USE_SCI == False: ## remove science data from DataFrame/Series
        ref_data = asm_data[colname].drop(index=isci).to_numpy() 
    else:
        ref_data = asm_data[colname].copy()

    if param == 'WDH':
        return wind_effects(ref_data, sci_data, cube_index, param_max)
    
    else:
        frame_select = np.zeros((len(sci_data),param_max), dtype=int)
        best_data = np.zeros_like(frame_select, dtype=float)
        cubes = []
        ## for each science frame, select param_max reference frames with smallest parameter difference
        for sci_framei, sci_val in enumerate(sci_data):
            delta_p = np.abs(ref_data-sci_val)
            frame_select[sci_framei] = np.argsort(delta_p)[:param_max]
            best_data[sci_framei] = ref_data[frame_select[sci_framei]]
            cubes += list(cube_index[frame_select[sci_framei]])
        
        if param == 'TAU0':
            sci_data*=1e3; best_data*=1e3
        ## print science values and number of cubes and range of values of preselection
        print('> Science frame values [{0:s}]: median = {1:.2f}, min = {2:.2f}, max = {3:.2f}'.format(
                P_UNIT[param], np.nanmedian(sci_data), np.nanmin(sci_data), np.nanmax(sci_data)))
        print('> Best matching reference values (from {0:d} targets): median = {1:.2f}, min = {2:.2f}, max = {3:.2f} \n'.format(
                len(set(cubes)), np.nanmedian(best_data), np.nanmin(best_data), np.nanmax(best_data)))
        
        return frame_select, set(cubes)

## manipulate a list of angles so they are between 0 and 360 deg
def correct_angle(angles):
    if type(angles) in [tuple, int, float]:
        angles = list(angles)
    for i,a in enumerate(angles):
        while a > 360:
            a -= 360
        while a < 0:
            a += 360
        angles[i] = a
    return angles

## difference between two angles, accounting for wrap at 360 deg !!angles must be between 0 and 360 deg
def angle_dif(a, b):
    a,b = correct_angle([a,b])
    c=abs(a-b)
    if c<180: return c
    else: return abs(360-c)

## preselection for 200mbar wind that takes both speed and on frame direction into account
def wind_effects(ref_data, sci_data, cube_index, param_max):
    frame_select = np.zeros((len(sci_data),param_max), dtype=int)
    best_data = np.zeros((sci_data.shape + (param_max,)))
    cubes = []
    
    for sci_framei, sci_val in enumerate(sci_data):
        delta_speed = np.abs(ref_data[:,0]-sci_val[0]) 
        delta_dir = np.apply_along_axis(angle_dif, 1, ref_data[:,1:2], b=sci_val[1])[:,0]
        
        ## need to consider both speed and direction, so normalise values between 0 and 1
        ## (or <1 if dir), then calculate RMS for each row
        norm = np.stack((delta_speed/np.nanmax(delta_speed), delta_dir/180), axis=1) 
        delta_p = np.sqrt(np.mean(np.square(norm),axis=1))
        frame_select[sci_framei] = np.argsort(delta_p)[:param_max]
        best_data[sci_framei] = ref_data[frame_select[sci_framei]].T
        cubes += list(cube_index[frame_select[sci_framei]])
    
    ## print science wind speed and number of cubes and range of wind speeds of preselection 
    print('> Science frame wind speed [m/s]: median = {0:.2f} , min = {1:.2f}, max = {2:.2f}'.format(
            np.nanmedian(sci_data[:,0]), np.nanmin(sci_data[:,0]), np.nanmax(sci_data[:,0])))
    print('> Best matching reference speed (from {0:d} targets): median = {1:.2f}, min = {2:.2f}, max = {3:.2f}'.format(
            len(set(cubes)), np.nanmedian(best_data[:,0]), np.nanmin(best_data[:,0]), np.nanmax(best_data[:,0])))
    ## print on frame wind direction of science, and difference in direction of references
    sci_dir = np.rad2deg(np.unwrap(np.deg2rad(sci_data[:,1])))
    print('> Science wind direction (on frame) [deg]: {0:.2f} - {1:.2f}'.format(*correct_angle([min(sci_dir),max(sci_dir)])))
    print('> Reference wind offset [deg]: median = {0:.2f}, min = {1:.2f}, max = {2:.2f} \n'.format(
            np.nanmedian(best_data[:,1]), np.nanmin(best_data[:,1]), np.nanmax(best_data[:,1])))

    return frame_select, list(set(cubes))

"""
--------------------------------------------------------------------------------
"""
## takes target data and performs preselection for each science frame or the full science cube
## and returns the frame or cube indices of the preselection
def preselection(p, target_table, isci, nframes, ncorr, param_max, USE_SCI):
    best_param = np.sort(per_target(target_table, isci, p, param_max, ncorr, nframes))
    return best_param