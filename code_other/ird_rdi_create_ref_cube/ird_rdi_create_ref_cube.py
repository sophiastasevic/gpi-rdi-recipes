"""
Created on Mon Apr 25 15:00:42 2022

@author: stasevis

Find best correlated reference frames for the science cube using correlation matrix and
save reference cube containing these frames

sof file containing: correlation matrix, science target, corr frame selection vector, 
reference table, reference asm data file, and frame selection vector [OPTIONAL]
--ncorr: no. best correlated frames per science frame to include in library
--crop: frame size of reference library output in pixels
--use_select_vect [OPTIONAL]: flag to use frame selection vector for science cube
--remove_science [OPTIONAL]: flag to not use science frames in reference library
--param [OPTIONAL]: preselect reference frames that most closly match the science frames for a
                    given parameter:
                    [SEEING, TAU0, DIT, NDIT, EPOCH, MG, MH, LWE, WDH, SPT, PCC, BEST, RAND]

outputs:
    'reference_cube*.fits'
    'target_info_table*.fits'
    'frame_selection_vector*.fits'

[2022-05-03] added option to crop reference frames
[2022-05-06] saves target + observation information of all data cubes to fits table
[2022-08-16] saves all wavelengths in single cube with added frame selection vector
[2023-04-23] compatability with frame selection vectors being used in correlation matrix
             to identify correct reference frames from cubes.
[2023-06-02] option to generate ref lib per science frame when score==1
[2023-07-04] additional header parameter based selection
[2023-11-09] seeing, tau0, epoch, and winds now have values per frame + removed option
             to create a single ref lib for the entire science cube (only per frame)
[2024-04-15] all reference target data stored in output table of correlation recipe
[2024-07-16] option for a best mix of parameters and random frame library
"""

import argparse
import warnings, os
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table, Column
from astropy.time import Time
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.utils.exceptions import AstropyWarning
import ref_lib_table as rtab
import ref_frame_preselection as preselect

from retry_handler import RetryHandler

warnings.simplefilter('ignore', category=AstropyWarning)

## colum names in asm_data csv or ref_table for each parameter
param_names = {'SEEING': 'seeing',
               'TAU0': 'tau0',
               'EPOCH': 'timestamp_mjd',
               'LWE': ['wind_speed_30m','wind_dir_30m_onframe'],
               'WDH': ['wind_speed_200mbar','wind_dir_200mbar_onframe'],
               'ELEV': 'elevation',
               'EXPT': 'Total_exposure_time',
               'DIT': 'Exptime',
               #'NDIT': 'NDIT',
               'MG': 'G_mag',
               'MH': 'H_mag',
               'SPT': 'SpT_decimal'}

## list of parameter to use when compiling a 'MIX' parameter library 
MIX_PARAM = ['SEEING', 'TAU0', 'DIT', 'ELEV', 'EPOCH', 'EXPT', 'MG', 'MH', 'SPT', 'LWE', 'WDH']

## possible parameters to define reference library preselection
PARAM_LIST = ['SEEING','TAU0','DIT','NDIT','EPOCH','MG','MH','LWE','WDH','ELEV','EXPT','SPT','MIX','PCC','RAND']

## parameters which may vary by frame, rather than being the same for a full observation
PER_FRAME = ['SEEING','TAU0','EPOCH','LWE','WDH','ELEV']

"""
---------------------------------- FUNCTIONS ----------------------------------
"""
def read_file(func, path, **kwargs):
    if "summer" in path:
        return RetryHandler(func).run(path, **kwargs)
    else:
        return func(path, **kwargs)
    
##add reference frame information to reference library header
def add_header_ref_info(cube_hdr, i, nframes, ref_hdr):
    ref_hdr['OBJ_{0:04d}'.format(i)] = cube_hdr['OBJECT']
    ref_hdr['RA_{0:04d}'.format(i)] = cube_hdr['RA']
    ref_hdr['DEC_{0:04d}'.format(i)] = cube_hdr['DEC']
    ref_hdr['OBS_{0:04d}'.format(i)] = cube_hdr['DATE-OBS']
    ref_hdr['DAT_{0:04d}'.format(i)] = cube_hdr['DATE']
    ref_hdr['N_{0:04d}'.format(i)] = nframes
    if i == 0:
        ref_hdr.comments['DAT_{0:04d}'.format(i)] = 'convert cube file created on this date'
        ref_hdr.comments['N_{0:04d}'.format(i)] = 'number of frames used from this cube'
    
    return ref_hdr

def add_frame_elevation(asm_data, ref_table, ref_range):
    def elevation(ra, dec, times, location="paranal", frame="fk5"):
        radec = SkyCoord(ra, dec, frame=frame, unit="deg")
        location = EarthLocation.of_site(location)
        altaz = radec.transform_to(AltAz(obstime=times, location=location))
        return altaz.alt.to_value(unit="deg")
    elev = np.zeros(len(asm_data))
    for i, framei in enumerate(ref_range):
        ra, dec = ref_table[i]['RA','Dec']
        times = Time(asm_data['timestamp_mjd'][framei[0]:framei[1]],format='mjd')
        elev[framei[0]:framei[1]] = elevation(ra, dec, times)
    return elev
 
## return list of reference frames for each science frame and wavelength, and frame selection vector
def sort_frame_list(best_corr, ref_corr, channels, nframes_sci):
    ## set of reference frames for all science frames     
    frame_index = list(set(np.reshape(best_corr,np.product(best_corr.shape))))
    frame_index.sort()
    
    ## frame selection vector specifying which reference frames belong to which science frame's/
    ## wavelength channels' reference library (avoid saving separate ref cube for each one)
    wl_frame_select = np.zeros((channels, nframes_sci, len(frame_index)), dtype=int)
    for  wli in range(channels):
        for isci in range(nframes_sci):
            for i,index in enumerate(frame_index):
                if index in best_corr[wli,isci]: wl_frame_select[wli,isci,i]=1
        
    ## 0:median, 1:min, 2:max, 3:lower quartile, 4:upper quartile
    stats = np.concatenate((np.stack((np.median(ref_corr), ref_corr.min(), ref_corr.max())), np.quantile(ref_corr,(0.25,0.75))))
    stats_str = ', '.join(['{0:s} = {1:.4f}'.format(x,y) for x,y in zip(['median','min','max','lower quartile','upper quartile'],stats)])
    print('> Frame correlation:', stats_str)
    
    return frame_index, wl_frame_select

## score correlation of reference frames to science frames
def score_frames(corr_matrix, ncorr, param_lim=None, prev_best=None): 
    """ finds best [ncorr] correlated reference frames for each science frame
        - OPT. PARAM [param_lim]: applies frame preselection
    """
    channels, nframes_sci, nframes_ref = corr_matrix.shape
    
    if param_lim is None: 
        param_lim = np.arange(nframes_ref) ## set of all ref frames
    npar_frame = param_lim.shape[-1] ## number of best parameter frames
    
    if prev_best is not None:
        print('> Finding best correlated frames from {0:d} reference frames.'.format(npar_frame))
    
    if npar_frame < ncorr:
        print('> [Warning] not enough well matching frames for requested ncorr, setting ncorr to {0:d}'.format(npar_frame))
        ncorr = npar_frame
    if param_lim.ndim == 1:
        best_param = param_lim ## to account for some parameter selection selection being "per frame" during for loop
    
    best_corr = np.zeros((channels, nframes_sci, ncorr), dtype=int)
    ref_corr = np.zeros((channels, nframes_sci, ncorr))
    for wli in range(channels):
        for i in range(nframes_sci):
            if param_lim.ndim == 2:
                best_param = param_lim[i]
            if prev_best is not None:
                best_param = np.array([x for x in best_param if x not in prev_best[(wli, i)]], dtype=int)
                                        ## -ve to account for nan values that would be put up front using [::-1]
            best_corr[wli,i] = np.argsort(-abs(corr_matrix[wli,i,best_param]))[:ncorr]
            best_corr[wli,i] = best_param[best_corr[wli,i]] ## get correct indices
            
            ref_corr[wli,i] = corr_matrix[wli,i,best_corr[wli,i]]
    
    if prev_best is not None:
        return best_corr, ref_corr
    else:
        return sort_frame_list(best_corr, ref_corr, channels, nframes_sci)

## finds the best correlated, unique reference frames for each parameter in MIX_PARAM
def best_mix_frames(corr_matrix, ncorr, best_param, science_hdr):
    round_up = lambda x: int(x) + (x-int(x)>0)
    param_ncorr = round_up(ncorr/len(best_param))
    ncorr_tmp = int(param_ncorr * len(best_param))
    
    channels, nframes_sci = corr_matrix.shape[:-1]
    best_corr = np.zeros((channels, nframes_sci, ncorr_tmp), dtype=int)
    ref_corr = np.zeros((channels, nframes_sci, ncorr_tmp))
    
    for i,p in enumerate(best_param.keys()):
        i0 = i*param_ncorr; i1 = (i+1)*param_ncorr
        res = score_frames(corr_matrix, param_ncorr, best_param[p], best_corr[...,:i0])
        best_corr[...,i0:i1] = res[0]
        ref_corr[...,i0:i1] = res[1]
        
    if ncorr != ncorr_tmp:
        print("> Could not split {0:d} frames evenly for {1:d} parameters. Removing least correlated excess frames.".format(
                ncorr,len(MIX_PARAM)))
        keep = np.argsort(-np.abs(ref_corr))[...,:ncorr]
        best_corr = np.take_along_axis(best_corr, keep, axis=-1)
        ref_corr = np.take_along_axis(ref_corr, keep, axis=-1)
        
    return sort_frame_list(best_corr, ref_corr, channels, nframes_sci)
 
## generate NCORR random, unique indices for reference frame selection
def random_frames(best_param, ncorr, channels, nframes_sci):
    randi = []
    count = 0
    while count < ncorr:
        randi = set(list(randi) + list(np.random.randint((len(best_param)), size=(ncorr - count))))
        count = len(randi)

    frame_index = np.sort(best_param[list(randi)])
    wl_frame_select = np.ones((channels, nframes_sci, ncorr), dtype=int)
    
    return frame_index, wl_frame_select


## returns indices of ref cubes containing selected reference frames + frame indices and scores
def update_ref_cubes(frame_index, corr_frames, nframes):  
    #corr_frames = np.array(corr_frames, dtype=int) 
    target_index = []
    ref = {'target_index':np.zeros(nframes, dtype=int), ##index of target in list of selected ref cubes
           'frame_index':np.zeros(nframes, dtype=int), ##index of frame in master cube
           'corr_target_index':np.zeros(nframes, dtype=int), ##index of target in list of ref cubes used in correlation matrix
           'corr_frame_index':np.zeros(nframes, dtype=int)} ##infex of frame in correlation matrix
    
    for i,frame in enumerate(frame_index):
        n,x = corr_frames[:,frame] #frame index in target cube and target no. in matrix
        if n not in target_index:
            target_index.append(n)
            
        ref['target_index'][i] = target_index.index(n)
        ref['frame_index'][i] = x
        ref['corr_target_index'][i] = n
        ref['corr_frame_index'][i] = frame

    return np.array(target_index), ref

## adds reference frames from reference cube + updates header
def make_ref_cube(ref, ref_path, nframes, crop, ref_hdr):
    ref_cube = []
    frame_count = 0
    ref_count = 0
    frame_info = rtab.init_frame_dict(nframes)
    problem_frames = []
    
    for refi, path in enumerate(ref_path):
        index = np.where(ref['target_index']==refi)[0]
        frames = ref['frame_index'][index]
        try:
            if read_file(os.path.exists, path) == False and "data_ext" in path:
                ##TODO: this check for "data_ext_cobrex" isn't in the dc vers
                if read_file(os.path.exists, path.replace("data_ext","data_ext_cobrex")) == True:
                    path = path.replace("data_ext","data_ext_cobrex")
                else:
                    path_tmp = path.split("data_ext")
                    for i in range(1,6):
                        if read_file(os.path.exists, path_tmp[0]+"data{0:d}_ext".format(i)+path_tmp[1]) == True:
                            path = path_tmp[0]+"data{0:d}_ext".format(i)+path_tmp[1]
                            break
            cube_tmp, hdr_tmp = read_file(fits.getdata, path, header=True)
            if crop is not None:
                x,y = cube_tmp.shape[-2:]
                if (x+crop)%2:
                    x+=1
                xmin, xmax = ((x-crop)//2, (x+crop)//2)
                cube_tmp = cube_tmp[...,xmin:xmax,xmin:xmax]

            ref_cube.append(cube_tmp[:,frames])
            ref_hdr = add_header_ref_info(hdr_tmp, ref_count, len(frames), ref_hdr)
            frame_count += len(frames)
            ref_count += 1

        except FileNotFoundError:
            print('> [Warning] File not found with path: {0:s}, \n\
                  \t.. Continuing to next reference target .. \n'.format(path))
            hdr_tmp = None
            if len(problem_frames) == 0:
                problem_frames = list(index)
            else:
                problem_frames += list(index)
        
        rtab.add_frame_info(frame_info, frame_count, ref_count-1, index, hdr_tmp)    
            
    ref_hdr.set('NCUBE', len(ref_cube), 'total number of ref targets used', before='OBJ_0000')
    ref_hdr.set('NFRAME', frame_count, 'total number of ref frames', after='NCUBE')
    ref_hdr.set('', '----------------', after='NFRAME')

    print('> Reference library built using {0:d} reference targets and {1:d} frames.'
          .format(len(ref_cube),frame_count))

    return np.concatenate(ref_cube, axis=1), ref_hdr, frame_info, problem_frames
    

def update_header(science_hdr, ncorr, use_select_vect, sci_corr, corr_hdr, param_max):
    science_hdr['NCORR'] = (ncorr, 'no. best correlated reference frames for each science frame')
    science_hdr['NPARAM'] = (param_max, 'no. well matching parameter frames used in preselection')
    science_hdr['PER_FRAME'] = True
    science_hdr['USE_SCI'] = sci_corr
    science_hdr['SEL_VECT'] = (use_select_vect, 'frame selection vector used for science cube')
    science_hdr['PARAM'] = (' ', 'parameter used to optimise reference library') #placeholder
    
    science_hdr['CORR_IN'] = (corr_hdr['INNER_R'], 'inner radius of correlation annulus')
    science_hdr['CORR_OUT'] = (corr_hdr['OUTER_R'], 'outer radius of correlation annulus')
    for wl in {0:[0], 1:[1], 2:[0,1]}[corr_hdr['WL_CHOSE']]:
        hname = 'WL{0:d}_AO_R'.format(wl)
        science_hdr[hname] = corr_hdr[hname]

def get_path(fnames, ftypes, frame):
    path = fnames[np.where(ftypes==frame)[0]]
    if len(path)!=1:
        raise Exception('[Error] Input must contain one', frame)
    
    return path[0]

def fix_header(path):
    hdul = read_file(fits.open, path)
    hdul.verify('fix')
    
    return hdul[0].header

#%%
"""
---------------------------------- MAIN CODE ----------------------------------
"""
parser = argparse.ArgumentParser()
parser.add_argument('sof', help='name of sof file', type=str)
parser.add_argument('--ncorr', help='no. of best correlated ref frames to include for each science frame', type=int, default=100)
parser.add_argument('--crop', help='px size to crop frames to', type=int)
parser.add_argument('--use_select_vect', action='store_true', help='use science frame selection vector for reference frame selection (when not used for correlation matrix)')
parser.add_argument('--remove_science', action='store_true', help='do not use science frames in reference library if in correlation matrix')
parser.add_argument('--param', help='ref target observation parameter to match to science, options: SEEING, TAU0, DIT, NDIT, EPOCH, MG, MH, LWE, WDH, SPT, PCC [comma separate]', type=str, default='PCC')
parser.add_argument('--param_max', help='number of reference frames to preselect for closely matching observing conditions', type=int, default=10000)
parser.add_argument('--mix_max', help='number of parameter subgroup reference frames to preselect for MIX library [if building multiple libraries]', type=int)

args = parser.parse_args()

sof = args.sof
ncorr = args.ncorr
crop = args.crop
use_select_vect = args.use_select_vect
remove_sci = args.remove_science
if (args.param).upper() == 'ALL':
    param = PARAM_LIST
else:
    param = [(p.strip()).upper() for p in (args.param).split(',')]
    for p in param:   
        if p not in PARAM_LIST:
            raise Exception('[Error] {0:s} is not a recognised parameter name, please select from: {1:s}'.format(p,', '.join(PARAM_LIST)))
param_max = args.param_max
if args.mix_max == None:
    mix_param_max = param_max
else:
    mix_param_max = args.mix_max

##getting file names of input data from sof file
data = np.loadtxt(sof,dtype=str)
fnames = data[:,0]
ftypes = data[:,1]

paths = {'science':'IRD_SCIENCE_REDUCED_MASTER_CUBE',
         'frame_select':'IRD_FRAME_SELECTION_VECTOR',
         'corr':'IRD_CORR_MATRIX',
         'corr_select':'IRD_CORR_FRAME_SELECTION_VECTOR',
         'ref_table':'IRD_REFERENCE_TABLE',
         'ref_asm':'IRD_REFERENCE_FRAME_DATA'}

for var in paths.keys():
    if var == 'frame_select' and use_select_vect == False:
        paths[var] = None
    else:
        paths[var] = get_path(fnames, ftypes, paths[var])
        
radius = {True:'_ao_ring', False:''}['ao_ring' in paths['corr']]
#%%
print('.. Reading in science data ..')

science_hdr = fix_header(paths['science'])
sci_nframes = science_hdr['NAXIS3']

frame_vect = read_file(fits.getdata, paths['corr_select'])
if frame_vect.shape[0] > 2:
    frame_vect=np.array(frame_vect[:2],dtype=int)

corr_matrix, corr_hdr = read_file(fits.getdata, paths['corr'], header=True)
ncube_sci = corr_hdr['NB_SCI_CUBES']
if ncube_sci > 1: ##correlation matrix for multipl science cubes
    if science_hdr['DATE-OBS'] not in corr_hdr['OBS_*'].values():
        raise Exception('[Error] Science cube not in correlation matrix science target list.')
        
    for isci in range(ncube_sci):
        if corr_hdr['OBS_{0:03d}'.format(isci)] == science_hdr['DATE-OBS']: break
    
    sci_frm0 = corr_hdr['FRM0_{0:03d}'.format(isci)]
    if isci == ncube_sci-1:
        sci_frm1 = corr_matrix.shape[1]
    else:
        sci_frm1 = corr_hdr['FRM0_{0:03d}'.format(isci+1)]
    corr_matrix = corr_matrix[:,sci_frm0:sci_frm1]
else:
    if science_hdr['DATE-OBS'] != corr_hdr['DATE-OBS']:
        raise Exception('[Error] Science cube not in correlation matrix science target list.')
    isci = 0

sci_ref_posn = [x for x in corr_hdr['REF_POSN*'].values()]    
sci_refi = sci_ref_posn[isci] #corr_hdr['REF_POSN_{0:03d}'.format(isci)] ##position of science cube in reference cube list
sci_corr_nframes = corr_matrix.shape[1]

wl_convert = {0: [0], 1: [1], 2: [0,1]}
wl_channel = wl_convert[corr_hdr['WL_CHOSE']]
inv_wl = {'[0]':wl_channel[0], '[1]':1, '[0 1]':2}


if use_select_vect and sci_nframes == sci_corr_nframes:
    frame_select = read_file(fits.getdata, paths['frame_select'])
    if len(frame_select.shape)>1:
        frame_select=frame_select[:,0]
    
    good_frames = np.where(frame_select==1.)[0]
    corr_matrix = corr_matrix[:,good_frames]
    sci_corr_nframes = len(good_frames)
    
elif sci_nframes > sci_corr_nframes:
    use_select_vect = True
    
print('> Reference frame selection using correlation of {0:d}(/{1:d}) science frames'.format(sci_corr_nframes, sci_nframes))

try:
    nref = int(corr_hdr["NB_REF_CUBES"])
except KeyError:
    nref = int(corr_hdr["ncube_ref"])

## if the index position of a science cube is larger than the number of ref cubes, then the science
## cube isn't in the correlation matrix and should not be considered during preselection
skip_ref = [i for i in sci_ref_posn if i>=nref]
## correct error in header frame count for science cubes that were not in the ref list
if ncube_sci > 1:
    skip_ref.sort()
    for i in skip_ref:
        sci_i = sci_ref_posn.index(i)
        corr_hdr["RF{0:06d}".format(i)] = corr_hdr['FRMS_{0:03d}'.format(sci_i)]
        corr_hdr["RS{0:06d}".format(i)] = corr_hdr["RS{0:06d}".format(i-1)] + corr_hdr["RF{0:06d}".format(i-1)]
if corr_matrix.shape[-1] < frame_vect.shape[-1]:
    skip_ref += list(set(frame_vect[0,corr_matrix.shape[-1]:]))
    
## cubes where the correlation is NaN are either other observations of the science target
## or different science targets in the matrix that have known signal, so should be skipped
skip_ref += list(set(frame_vect[0,np.where(~np.isfinite(corr_matrix[0,0]))[0]]))

##storing information about the reference targets used
ref_path_tmp = [x for x in corr_hdr["RN*"].values()]
ref_nframes = np.array([x for x in corr_hdr["RF*"].values()],dtype=int)
ref_start = np.array([x for x in corr_hdr["RS*"].values()],dtype=int)
ref_range = np.stack((ref_start, ref_start+ref_nframes), axis=1)
    
#param_max = int(np.sum(ref_nframes)*MAX_FRACT)

##TODO: try to get this to work if corr_matrix run on SPHERE-DC, but if it doesn't work,
##      need to do a raise Exception earlier in the code if the process is being run on 
##      the grid but the ref cube paths are on the server
if "summer" not in paths['science'] and "summer" in ref_path_tmp[0]:
    ref_path_tmp = ["/dwh/"+rpath.split('/dwh/')[-1] for rpath in ref_path_tmp]
elif "summer" in paths['science'] and "summer" not in ref_path_tmp[0]:
    ref_path_tmp = ["/summer/sphere/data_ext"+rpath for rpath in ref_path_tmp]

for refi, path in enumerate(ref_path_tmp):
  if read_file(os.path.exists, path) == False: 
      if "data_ext" in path:
          if read_file(os.path.exists, path.replace("data_ext","data_ext_cobrex")) == True:
              ref_path_tmp[refi] = path.replace("data_ext","data_ext_cobrex")
          else:
              path_tmp = path.split("data_ext")
              add_to_skip = True
              for i in range(1,6):
                  if read_file(os.path.exists, path_tmp[0]+"data{0:d}_ext".format(i)+path_tmp[1]) == True:
                      ref_path_tmp[refi] = path_tmp[0]+"data{0:d}_ext".format(i)+path_tmp[1]
                      add_to_skip = False
                      break
              if add_to_skip == True:
                  skip_ref.append(refi)
      else:
          skip_ref.append(refi)

try:
    sci_corr = corr_hdr['USE_SCI']
except KeyError:
    sci_corr = paths['science'] in ref_path_tmp

if remove_sci == True and sci_corr == True:
    corr_matrix[...,ref_range[sci_refi,0]:ref_range[sci_refi,1]] = np.nan
    sci_corr = False #science frames not used in reference library
    skip_ref.append(sci_refi)

skip_ref = list(set(skip_ref))
skip_ref.sort()

update_header(science_hdr, ncorr, use_select_vect, sci_corr, corr_hdr, param_max)

asm_data = pd.read_csv(paths['ref_asm'], index_col=0)
ref_table = Table.read(paths['ref_table']) 

## compute frame elevation
asm_data.insert(0, param_names['ELEV'], add_frame_elevation(asm_data,ref_table,ref_range))

## compute total exposure time
ref_table.add_column(Column(ref_table['Exptime']*ref_table['Cube_Nframes'], name=param_names['EXPT']))
  
print('> Finding best correlated frames from {0:d} reference target cube(s)'.format(nref))

for p in param:
    print('\n.. Creating reference library with best matching {0:s} frames ..'.format(p))
    #if p == 'BEST':
    #    science_hdr['PARAM'] = 'MIX'
    #else:
    science_hdr['PARAM'] = p
    
    if p == 'MIX':
        ## selection of best correlated reference frames for each parameter in MIX_PARAM
        best_param = {}
        for p_tmp in MIX_PARAM:
            print('{0:s}:'.format(p_tmp))
            if p_tmp in PER_FRAME:
                ## observational conditions which vary by frame
                best_param[p_tmp] = preselect.per_frame(asm_data, sci_refi, p_tmp, param_names[p_tmp], frame_vect[0], skip_ref, mix_param_max)  
            else:
                ## stellar parameters and set observation parameters (DIT, NDIT) which are constant for a given observation
                best_param[p_tmp] = preselect.per_target(ref_table, sci_refi, p_tmp, param_names[p_tmp], frame_vect[0], skip_ref, mix_param_max, ncorr, ref_nframes)
        frame_index, wl_frame_select = best_mix_frames(corr_matrix, ncorr, best_param, science_hdr)
    
    elif p == 'RAND':
        ## random reference library
        ## all frames that are not in skip_ref (i.e. are finite in the correlation matrix)
        best_param = np.where(np.isfinite(corr_matrix[0,0]))[0]
        frame_index, wl_frame_select = random_frames(best_param, ncorr, len(wl_channel), sci_corr_nframes)
    
    else:
        if p == 'PCC':
            ## all frames that are not in skip_ref (i.e. are finite in the correlation matrix)
            best_param = np.where(np.isfinite(corr_matrix[0,0]))[0]
        elif p in PER_FRAME:
            ## observational conditions which vary by frame
            best_param = preselect.per_frame(asm_data, sci_refi, p, param_names[p], frame_vect[0], skip_ref, param_max)  
        else:
            ## stellar parameters and set observation parameters (DIT, NDIT) which are constant for a given observation
            best_param = preselect.per_target(ref_table, sci_refi, p, param_names[p], frame_vect[0], skip_ref, param_max, ncorr, ref_nframes)
        frame_index, wl_frame_select = score_frames(corr_matrix, ncorr, best_param)
    
    if p == 'MIX':
        science_hdr['NPARAM'] = mix_param_max
    else:
        science_hdr['NPARAM'] = best_param.shape[-1]
    
    nframes = len(frame_index)
    target_index, ref = update_ref_cubes(frame_index, frame_vect, nframes)
    ## paths for cubes with frames selected for library
    ref_path = [ref_path_tmp[i] for i in target_index]
        
    if len(wl_frame_select.shape)<3:
        frame_channel = np.zeros(nframes, dtype=int)
        for n in range(nframes):
            wl = np.where(wl_frame_select[:,n]>0)[0]
            frame_channel[n] = inv_wl[str(wl)]
    else: frame_channel = None
        
    ref_cube, ref_hdr, frame_info, problem_frames = make_ref_cube(ref, ref_path, nframes, crop, science_hdr.copy())
    frame_table = rtab.make_frame_table(frame_info, ref, problem_frames, sci_refi, frame_channel)
    
    table_hdu = fits.BinTableHDU(frame_table, header=ref_hdr)
    table_hdu.writeto('frame_info_table_{0:s}{1:s}.fits'.format(p,radius))
        
    frame_selection_vector = np.delete(wl_frame_select, problem_frames, axis=-1)
    frame_hdu = fits.PrimaryHDU(data = frame_selection_vector, header = ref_hdr)
    frame_hdu.writeto('frame_selection_vector_{0:s}{1:s}.fits'.format(p,radius))
        
    hdu = fits.PrimaryHDU(data = ref_cube, header = ref_hdr)
    hdu.writeto('reference_cube_{0:s}{1:s}.fits'.format(p,radius))
