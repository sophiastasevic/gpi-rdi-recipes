# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 14:34:08 2022

@author: sophi
"""
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
import os, sys, re
import numpy as np
import pandas as pd
import subprocess
from astropy import units as u
import get_star_info as simbadtool

D_TEL = 8.2 #/m
S_DLIM = u.rad.to(u.arcsec,1.22*500e-9/D_TEL) #SPARTA diffraction limit /arcsec

## prevents calls to print when calling a function using 'with HiddenPrints():'
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        
"""
--------------------------------- FUNCTIONS ---------------------------------
"""
## set all missing string values to '' and remove multiple spaces
clean_str = lambda name: re.sub(r'\s+',' ',name.strip())
def clean_string(t, col):
    for i,val in enumerate(t[col]):
        if type(val) not in [str, np.str_, np.string_] or val == '--':
            t[col][i] = ''
        elif re.search(r'\s\s+',val) is not None:
            t[col][i] = clean_str(val)
        else:
            t[col][i] = val.strip()
            
## creating fits table containing information for each data cube, including
## science, reference, and unused reference targets
def add_target_info(target_info, columns, refi, sid, ref_hdr=None, cache=None):
    if ref_hdr is not None:
        get_target_info(target_info, columns, ref_hdr, refi, sid)
    #else:
    #    for k in columns.loc[columns['data_type']=='str'].index:
    #        target_info[k].append('')

## initialise dict for storing data cube information
def init_info_dict(ncube, columns, cube_paths=None):
    target_info = {k:np.zeros(ncube,dtype=columns.at[k,'data_type']) 
                   if columns.at[k,'data_type']!='str' 
                   else ['' for i in range(ncube)] for k in columns.index}     
    return target_info

## query COBREX star service using coords + return output csv as a pandas dataframe
def query_star(ra, dec):
    filename = 'star_data.csv'
    url = "http://cobrex-dc.osug.fr:8080/star-server/api/fcoo/csv/d%20{0:f}%20{1:f}".format(ra,dec)
    
    request_star_str = ['wget', '-O', filename, url]
    output,error = subprocess.Popen(request_star_str, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()
    df = pd.read_csv(filename, sep=';', skiprows=0, skipfooter=0, parse_dates=True, quoting=3)
    
    binary = False
    ## more than one star identified, flag as binary if distance between them is
    ## smaller than half the IRDIS FOV (assumes one star is centered)
    if df.shape[0] > 1:
        ## only difference between first two objects (if > 2) since this is the smallest sep
        sep = df['dist[deg]'].diff()[1]*3600 ##arcsec
        if sep < (512*0.01225):
            binary = True
            if pd.isna(df.loc[0,'flux g']) and pd.notna(df.loc[1,'flux g']):
                return df.loc[1].to_dict(), binary
    
    return df.loc[0].to_dict(), binary


## query COBREX star service for object by coord, if this fails, use simbad query 
def star_info(ra, dec, known_signal, data, name, date, columns):
    query, binary = query_star(ra, dec)
    if query['sphere id'] == -1:
        with HiddenPrints(): sim = simbadtool.get_star_info(name, date, ra, dec)
        if sim is None:
            with HiddenPrints(): sim = simbadtool.get_star_info(None, date, ra, dec)
            
        for k in columns.loc[columns['source']=='star'].index:
            query[columns.loc[k,'star_name']] = sim[columns.loc[k,'simbad_name']]
    else:
        query['spectral type'] = query['spectral type'].replace(' ~','')
    query['simbad main identifier'] = clean_str(query['simbad main identifier'])
    
    sid = query['simbad main identifier']
    if data is None or sid not in list(data['simbad main identifier']):
        for col in known_signal.keys():
            if sid in known_signal[col] or clean_str(sid) in known_signal[col]:
                query[col] = True
            elif col == 'binary':
                query[col] = binary
                sid_tmp = sid
                while sid_tmp[-1] in ['A','B','C','b']:
                    sid_tmp = sid_tmp[:-1].strip()
                if sid_tmp in known_signal[col] or clean_str(sid_tmp) in known_signal[col]:
                    query[col] = True
            else:
                query[col] = False
        if data is None:
            return sid, pd.Series(query).to_frame().T
        else:
            return sid, pd.concat([data, pd.Series(query).to_frame().T], ignore_index=True)
    else:
        return sid, data       


def complete_star_info(target_info, columns, star_data):
    for i,sid in enumerate(target_info['Simbad_ID']):
        for k in columns.loc[columns['source']=='star'].index:
            if k == 'Simbad_ID':
                continue ## already filled
            col, dtype = columns.loc[k,['star_name','data_type']]
            try:
                target_info[k][i] = star_data.at[sid,col]
            except (KeyError, TypeError):
                if dtype != 'str': target_info[k][i] = np.nan
                #else: target_info[k].append('')

            
## updating dict with information for each data cube
def get_target_info(target_info, columns, hdr, i, sid):
    name = hdr['OBJECT']
    if name in set(['No name','OBJECT,ASTROMETRY']):
        if name == hdr['ESO OBS TARG NAME']: name = hdr['ESO OBS NAME']
        else: name = hdr['ESO OBS TARG NAME']
    target_info['Object'][i] = name
    target_info['Simbad_ID'][i] = sid
    
    #star_info(target_info, i, hdr['RA'], hdr['DEC'], name, hdr['DATE-OBS'])
    
    for k in columns.loc[columns['source']=='hdr'].index:
        col, dtype = columns.loc[k,['hdr_name','data_type']]        
        #if dtype == 'str': target_info[k].append(hdr[col])
        target_info[k][i] = hdr[col]
    
    for k in columns.loc[columns['source']=='hdr_try'].index:
        try:
            target_info[k][i] = hdr[columns.at[k,'hdr_name']]
        except (KeyError, TypeError, fits.verify.VerifyError):
            target_info[k][i] = np.nan

    #try:
    #    target_info['Waffle_on'] = ('O' not in hdr['ESO OCS OBSTYPE LIST'])
    #except KeyError:
    #    target_info['Waffle_on'] = True
 
## flags first observation of an object        
def first_occurance(ref_table):
    ref_table.add_index('Simbad_ID')
    first = np.zeros(len(ref_table), dtype=int)
    sids = set(ref_table['Simbad_ID'])
    if '' in sids: sids.remove('')
    for sid in sids:
        i = ref_table.loc_indices[sid]
        if type(i) != list:
            first[i] = 1
        else:
            obs = list(ref_table['Obs_date'][i])
            first[i[obs.index(min(obs))]] = 1
    
    ref_table.remove_indices('Simbad_ID')
    ref_table['First_occurance'] = first
    
def format_spt(ref_table):
    spt_char = {'O':0,'B':10,'A':20,'F':30,'G':40,'K':50,'M':60}
    spt_lum = {'I':1, 'II':2, 'III':3, 'IV':4, 'V':5, 'VI':6, 'VII':7, 'd':6, 'D':7}

    ## match string with '+' followed by valid spectral class/'D'
    p_binary = re.compile(r'(\+(?:{0:s}))'.format('|'.join(list(spt_char.keys())+['D'])))
    
    ## match numeric component of spectral type
    p_num = re.compile(r'([0-9]+(?:\.[0-9]+)?)')
    
    ## match spectral class
    p_class = re.compile(r'{0:s}'.format('|'.join(list(spt_char.keys()))))
    
    ## match luminosity class
    p_lum = re.compile(r'I|V|D|d')
    
    ## match chemical composition elements appearing in spectral class
    p_FeC = re.compile(r'([A-Z][a-zA-Z](?:\-|\+)?[0-9]+(?:\.[0-9]+)?)')
    
    ## convert the spectral class (letters and numbers) into a decimal value
    ## for uncertain classifications (e.g. M4-6, B9/A0) mean value is taken
    def spt_decimal(spt):
        spt_class = [spt_char[x] for x in p_class.findall(spt)]
        if any([x.isdigit() for x in spt]):
            digit = np.mean([float(x) for x in p_num.findall(spt)])
        elif len(spt_class) == 1: ## only letter spectral class --> take middling value
            digit = 4.5
        else:
            digit = 0
        return np.mean(spt_class)+digit
    
    ## convert luminosity class into a decimal value
    def lum_decimal(spt):
        if len(p_lum.findall(spt)) == 0: ##no luminosity class: assume MS
            return spt_lum['V']
        else:
            lum = []; end = -1
            for i,l in enumerate(p_lum.finditer(spt)):
                start = l.start() ##starting position of match in full string
                if i == 0: ##first match
                    lum_tmp = l.group(0)
                elif start == end: ##match direct after previous one: same class
                    lum_tmp += l.group(0)
                else: ##new match
                    lum.append(lum_tmp)
                    lum_tmp = l.group(0)
                end = l.end()
            lum.append(lum_tmp)
            return np.mean([spt_lum[x] for x in lum])
    
    clean_string(ref_table,'SpT')
    ncubes = len(ref_table)
    spt_index = ref_table.index_column('SpT')
    for col in ['Lumi_decimal','SpT_decimal']:
        ref_table.add_column(np.zeros(ncubes), index=spt_index+1, name=col)
    
    spt_convert={}
    for i, (spt,obj) in enumerate(zip(ref_table['SpT'],ref_table['Simbad_ID'])):
        if type(spt) not in [str, np.str_, np.string_] or spt == '': ## no spectral type
            ref_table['SpT_decimal'][i] = np.nan
            ref_table['Lumi_decimal'][i] = np.nan
        else:
            if p_binary.search(spt) is not None:
                spt_tmp = [x for x in spt.split('+') if p_class.search(x) is not None]
                if len(spt_tmp)>1 and obj[-1].upper()=='B':
                    spt = spt_tmp[1]
                else:
                    spt = spt_tmp[0]
            spt = p_FeC.sub('',spt)
            try:
                ref_table['SpT_decimal'][i],ref_table['Lumi_decimal'][i] = spt_convert[spt]
            except KeyError:
                ref_table['SpT_decimal'][i] = spt_decimal(spt)
                ref_table['Lumi_decimal'][i] = lum_decimal(spt)
                spt_convert[spt] = (ref_table['SpT_decimal'][i],ref_table['Lumi_decimal'][i])           
 

    
def make_ref_table(target_info, columns, star_data, problem_stars=[]):
    complete_star_info(target_info, columns, star_data)
    
    colnames = [k for k in columns.index]        
    outside_range = lambda arr, lim_l, lim_u: np.array(list(set(np.where(arr<lim_l)[0])|
                                                            set(np.where(arr>lim_u)[0])))
    for col in [x for x in colnames if 'tel' in x or 'DIMM' in x]:
        target_info[col][np.where(target_info[col]==-1.)[0]] = np.nan
    
    for col in [x for x in colnames if 'SPARTA' in x]:
        target_info[col][np.where(target_info[col]==-1.)[0]] = np.nan
        if 'Strehl' in col and (any(target_info[col]>1) or any(target_info[col]<0)): 
            target_info[col][outside_range(target_info[col],0,1)] = np.nan
        if 'FWHM' in col and (any(target_info[col]>5) or any(target_info[col]<S_DLIM)): 
            target_info[col][outside_range(target_info[col],S_DLIM,5)] = np.nan
    
    ref_table = Table(target_info, names=colnames)
    for col in colnames:
        if type(columns.loc[col,'unit']) == str:
            ref_table[col].unit = columns.loc[col,'unit']
    
    if len(problem_stars)>0:
        ref_table.remove_rows(problem_stars) ##removes rows where ref target file not found
    
    #for col in ['SpT','Simbad_ID']: 
    #    clean_string(ref_table,col)
        
    epoch = Time(ref_table['Obs_date'])
    ref_table['Epoch_decimalyear'] = epoch.decimalyear * u.yr
    
    first_occurance(ref_table)
    format_spt(ref_table)
    
    return ref_table