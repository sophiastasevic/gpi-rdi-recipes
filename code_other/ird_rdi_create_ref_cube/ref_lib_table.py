#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 10:42:13 2023

@author: sstas
"""
import warnings
import numpy as np
from astropy.table import Table
from astropy.utils.exceptions import AstropyWarning

warnings.simplefilter('ignore', category=AstropyWarning)

"""
--------------------------------- TABLE FUNCTIONS ---------------------------------
"""
def add_frame_info(frame_info, frame_count, ref_count, index, hdr_tmp):
    index_mod = max(index) - frame_count + 1 ##if there are ref frames missing
    for i in index:
        if hdr_tmp is None:
            frame_info['Object'].append('-')
            frame_info['Obs_date'].append('-')
        else:
            frame_info['Object'].append(hdr_tmp['OBJECT'])
            frame_info['Obs_date'].append(hdr_tmp['DATE-OBS'])
            frame_info['RA'][i] = hdr_tmp['RA']
            frame_info['Dec'][i] = hdr_tmp['DEC']
            frame_info['Ref_frame'][i] = i - index_mod
            frame_info['Ref_target'][i] = ref_count
            
    return frame_info


def init_frame_dict(nframes):
    
    frame_info = {'Ref_frame':np.zeros(nframes,int),
                  'Ref_target':np.zeros(nframes,int),
                  'Object':[],
                  'RA':np.zeros(nframes),
                  'Dec':np.zeros(nframes),
                  'Obs_date':[]}
        
    return frame_info


def make_frame_table(frame_info, ref, problem_frames, sci_refi, frame_channel):#, param):
    
    colnames = ['Ref_frame','Ref_target','Object','Cube_frame',
                'Obs_date','RA','Dec','Corr_frame','Corr_target']
    
    frame_info['Corr_frame'] = ref['corr_frame_index']
    frame_info['Corr_target'] = ref['corr_target_index']
    frame_info['Cube_frame'] = ref['frame_index']
    
    if frame_channel is not None:
        colnames += ['Channel']
        frame_info['Channel'] = frame_channel
    
    frame_table = Table(frame_info, names=colnames)
    
    if len(problem_frames)>0:
        frame_table.remove_rows(problem_frames)
    
    if sci_refi in set(frame_table['Corr_target']):
        frame_table.add_index('Corr_target')
        frame_table['Science_frame'] = np.zeros(len(frame_table),dtype=bool)
        frame_table.loc[sci_refi]['Science_frame'] = True
    
    return frame_table
