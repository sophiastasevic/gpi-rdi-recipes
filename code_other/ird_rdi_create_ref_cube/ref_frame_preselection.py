#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 10:51:47 2023

@author: sstas
"""
import numpy as np

P_UNIT = {'SEEING': 'arcsec',
          'TAU0': 's',
          'EPOCH': 'mjd',
          'LWE': 'm/s',
          'WDH': 'deg',
          'ELEV': 'deg',
          'EXPT': 's',
          'PARANG': 'deg',
          '30M_SPEED': 'm/s',
          '30M_DIR': 'deg',
          '200MBAR_SPEED': 'm/s',
          '200MBAR_DIR':'deg'}

def ndit(delta_p, skip_ref, ncorr, ref_nfrms):
    rm_ref_tmp = np.ones(len(delta_p),dtype=bool)
    rm_ref_tmp[np.array(skip_ref)] = 0
    
    delta_tmp = delta_p[rm_ref_tmp]
    delta_set = np.sort(np.array(list(set(delta_tmp))))
    nfrms = np.array([ref_nfrms[rm_ref_tmp][np.where(delta_tmp<=i)[0]].sum() for i in delta_set])
    
    delta_min = delta_set[np.argmin(np.abs(nfrms-ncorr))]
    if delta_min <= 1:
        delta_min = 1
    else:
        print("[Warning] setting NDIT threshold to {0:.1f} offset with the science target.".format(delta_min))
    
    cutoff = len(np.where(delta_p<=delta_min)[0])
    return np.argsort(delta_p)[:cutoff]

    
def per_target(ref_table, isci, param, colname, frame_vect, skip_ref, param_max, 
               ncorr, ref_nframes, return_stats=False, return_val=False):
    ref_params = np.array(ref_table[colname])
    sci_param = ref_params[isci]
    delta_p = np.abs(ref_params-sci_param)
    
    if return_val == True:
        delta_p_full = np.zeros_like(frame_vect)
        for iref,r in enumerate(delta_p):
            fi = np.where(frame_vect==iref)[0]
            delta_p_full[fi] = r
        return delta_p_full 
    
    ## discreet values, find all in same group
    if param == 'DIT':
        param_sort = np.where(delta_p == 0)[0]
    elif param == 'NDIT':
        param_sort = ndit(delta_p, skip_ref, ncorr, ref_nframes)
        
    ## continuous values, find set number of best matching
    else:
        param_sort = np.argsort(delta_p) ## sort by smallest difference, i.e. closest match
        
    ## need to limit spectral type selection to be of the same luminosity class
    if param == 'SPT':
        colname = colname.replace('SpT','Lumi')
        sci_tmp = ref_table[colname][isci]
        ref_tmp = np.array(ref_table[colname])
        delta_tmp = np.abs(ref_tmp-sci_tmp)
        
        class_match = np.where(delta_tmp<1)[0]
        if class_match.size<param_max:
            class_match = np.where(delta_tmp<=1)[0]
            
        param_sort = np.array([x for x in param_sort if x in class_match])
            
    ## remove science target index from selection if not using science frames in ref library 
    ## as well as any other flagged reference targets
    for i in skip_ref:
        param_sort = param_sort[param_sort!=i]
            
    ## changing from reference target index to frame indices of target in correlation matrix
    best_match = np.zeros(param_max,dtype=int)
    count = 0
    target_index = 0
    while count < param_max:
        if target_index >= len(param_sort): ##not enough cubes, break early
            best_match = best_match[:count]; break
            
        iref = param_sort[target_index]
        fi = np.where(frame_vect==iref)[0]
        nfi = len(fi)
        if count+nfi > param_max:
            fi = fi[:param_max-count]    
            
        best_match[count:count+nfi] = fi ## don't need to recalc nfi since it will just fill the remaining cells
        count += nfi
        target_index += 1
        
    best_params = ref_params[param_sort[:target_index]]
    stats = target_stats(sci_param, best_params)
    
    if param == 'DIT':
        print('> Science value = {0:.1f}{1:s}; no. best matching frames = {2:d} (from {3:d} targets) \n'.format(
            sci_param, ref_table[colname].unit, len(best_match), target_index))        
    elif param == 'NDIT':
        print('> Science value = {0:d}; no. best matching frames = {1:d} (from {2:d} targets) \n'.format(
            int(sci_param), len(best_match), target_index))        
    else:
        if param == 'SPT':
            print('> Science value = {0:.2f} (lum class = {1:.2f}); no. best matching frames = {2:d} (from {3:d} targets)'.format(
                sci_param, sci_tmp, len(best_match), target_index))
            stats = np.concatenate((stats, target_stats(sci_tmp, ref_tmp[param_sort[:target_index]])))
        else:
            print('> Science value = {0:.4f}; no. best matching frames = {1:d} (from {2:d} targets)'.format(
                sci_param, len(best_match), target_index))
        print('> Best matching reference values: median = {0:.4f}, max = {1:.4f}, min = {2:.4f} \n'.format(
            np.nanmedian(best_params), np.nanmax(best_params), np.nanmin(best_params)))
    
    if return_stats == True:
        return best_match, stats
    else:
        return best_match


def frame_match(delta_p, flagged_frames, param_max):
    param_sort = np.argsort(delta_p)
    if len(flagged_frames)>0:
        rm_frm = np.ones(len(param_sort),dtype=bool)
        rm_frm[[np.where(param_sort==x)[0][0] for x in flagged_frames]] = False
        param_sort = param_sort[rm_frm]
    
    return param_sort[:param_max]


def per_frame(asm_data, isci, param, colname, frame_vect, skip_ref, param_max, parang=None, return_stats=False, return_val=False):
    flagged_frames = np.array((),dtype=int)
    for i in skip_ref:
        flagged_frames = np.append(flagged_frames,np.where(frame_vect==i)[0])
    
    sci_index = np.where(frame_vect==isci)[0]
    nsci = len(sci_index)
    nref = len(frame_vect)
    
    if param == 'WDH':
        return wind_effects(asm_data, sci_index, param, colname, flagged_frames, nsci, frame_vect, return_stats, return_val, param_max)
    
    elif param == 'LWE':
        ref_params = asm_data[colname[0]].to_numpy() ##colname[1] is wind direction
    
    elif param == 'PARANG':
        delta_p = np.zeros((nsci,nref))
        for i,s in enumerate(sci_index):
            for j,r in enumerate(parang):
                delta_p[i,j] = angle_dif(parang[s],r)
        if return_val == True:
            return delta_p
        ref_params = parang
    
    else:
        ref_params = asm_data[colname].to_numpy()
    
    if return_val == True:
        if 'DIR' in param or '200MBAR' in param:
            if 'DIR' in param: 
                colname = [colname[:-8].replace('dir','speed'),colname]
            else: 
                colname = [colname]
            return wind_effects(asm_data, sci_index, param, colname, flagged_frames, nsci, frame_vect, return_stats, return_val, param_max)
        else:
            delta_p_full = np.zeros((nsci,nref))
            for i,s in enumerate(sci_index):
                delta_p_full[i] = np.abs(ref_params - ref_params[s])
            return delta_p_full
    
    else:
        best_match = np.zeros((nsci,param_max),dtype=int)
        for i,s in enumerate(sci_index):
            if param != 'PARANG':
                delta_p = np.abs(ref_params - ref_params[s])
                best_match[i] = frame_match(delta_p, flagged_frames, param_max)
            else:
                best_match[i] = frame_match(delta_p[i], flagged_frames, param_max)
        
    sci_params = ref_params[sci_index]
    index_tmp = list(set(np.concatenate(best_match)))
    best_params = ref_params[index_tmp]
    nref_cube = len(set(frame_vect[index_tmp]))
    print('> Science frame values: median = {0:.4f} {1:s}, min = {2:.4f}, max = {3:.4f}'.format(
        np.median(sci_params), P_UNIT[param], min(sci_params), max(sci_params)))
    print('> Best matching reference values (from {3:d} targets): median = {0:.4f}, max = {1:.4f}, min = {2:.4f} \n'.format(
        np.nanmedian(best_params), np.nanmax(best_params), np.nanmin(best_params), nref_cube))
    
    if return_stats == True:    
        return best_match, frame_stats(sci_params, best_params)
    else:
        return best_match


## difference between two angles accounting for wrap at 360deg !!only if angles between 0 and 360 deg
def angle_dif(a, b):
    c=abs(a-b)
    if c<180: return c
    else: return abs(360-c)
    
def wind_effects(asm_data, sci_index, param, colname, flagged_frames, nsci, frame_vect, return_stats, return_val, param_max, get_dir=False):
    ref_speed = asm_data[colname[0]].to_numpy()
    
    if param == 'WDH' or 'DIR' in param:
        ref_dir = asm_data[colname[1]].to_numpy()
        get_dir = True
    
    if return_val == True:
        nref = len(ref_speed)
        delta_p = np.zeros((nsci,nref))
        for i,s in enumerate(sci_index):
            if get_dir == True:
                for j,r in enumerate(ref_dir):
                    delta_p[i,j] = angle_dif(ref_dir[s],r)
            else:
                delta_p[i] = np.abs(ref_speed - ref_speed[s])    
        return delta_p
    
    best_match = np.zeros((nsci,param_max),dtype=int)
    best_dir = np.zeros_like(best_match, dtype=float)
    
    for i,s in enumerate(sci_index):
        delta_speed = np.abs(ref_speed - ref_speed[s])
        
        delta_dir = np.zeros(len(ref_dir))
        for j,r in enumerate(ref_dir):
            delta_dir[j] = angle_dif(ref_dir[s],r)
        
        norm_speed = delta_speed/max(delta_speed)
        norm_dir = delta_dir/max(delta_dir)
        delta_p = np.sqrt(np.mean((norm_speed**2,norm_dir**2),axis=0))
        best_match[i] = frame_match(delta_p, flagged_frames, param_max)
        
        best_dir[i] = delta_dir[best_match[i]]
            
    sci_speed = ref_speed[sci_index]
    index_tmp = list(set(np.concatenate(best_match)))
    best_speed = ref_speed[index_tmp]
    nref_cube = len(set(frame_vect[index_tmp]))
    
    print('> Science wind speed [m/s]: median = {0:.4f} , min = {1:.4f}, max = {2:.4f}'.format(
        np.median(sci_speed), min(sci_speed), max(sci_speed)))
    print('> Best matching reference speed (from {3:d} targets): median = {0:.4f}, max = {1:.4f}, min = {2:.4f}'.format(
        np.nanmedian(best_speed), np.nanmax(best_speed), np.nanmin(best_speed), nref_cube))          
    
    sci_dir = np.rad2deg(np.unwrap(np.deg2rad(ref_dir[sci_index])))
    print('> Science wind direction [deg]: {0:.4f} - {1:.4f}'.format(min(sci_dir), max(sci_dir)))
    print('> Reference wind offset [deg]: median = {0:.4f} (max = {1:.4f}) \n'.format(np.nanmedian(best_dir), np.nanmax(best_dir)))

    if return_stats == True:
        stats = np.concatenate((frame_stats(sci_speed, best_speed), frame_stats(sci_dir, best_dir)))
        return best_match, stats
    else:
        return best_match


def frame_stats(sci_param, best_param):
    stats = np.array((np.nanmedian(sci_param), 
                      np.nanmin(sci_param), 
                      np.nanmax(sci_param)))
    return np.concatenate((stats, ref_param_stats(best_param)))

def target_stats(sci_param, best_param):
    stats = np.full(3, np.nan)
    stats[0] = sci_param
    return np.concatenate((stats, ref_param_stats(best_param)))
    
def ref_param_stats(best_param):
    stats = np.array((np.nanmedian(best_param), 
                      np.nanmin(best_param),
                      np.nanmax(best_param)))
    return stats