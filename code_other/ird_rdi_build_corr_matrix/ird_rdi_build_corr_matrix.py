"""
@AUTHOR: Sophia Stasevic
@DATE: 19/04/22
@CONTACT: sophia.stasevic@obspm.fr

Frame correlation between IRD_SCIENCE_REDUCED_MASTER_CUBE data
Adapted from recipe by Yuchen Bai; yuchenbai@hotmail.com [17/07/2021]

[SS] 20/04/22: Updated execution of loop to read in + calculate correlation one 
               cube at a time.
               
[SS] 25/04/22: Subtracts spatial mean from frames before calculating correlation.

[SS] 11/07/22: Added option to include science cube in list of reference cubes +
               output a fits file with observational parameters + mean correlation
               of the reference targets.
               
[SS] 21/10/22: Parameter table output and correlation of AO correction ring now
               options and input. Only mean of correlation annulus subtracted.

[SS] 27/04/23: Includes frame selection vectors of reference cubes + outputs the 
               concatenated selection.

[SS] 19/05/23: More robust checking for dublicate science target observations.
               
[SS] 21/05/23: Fixed bug in matching data cube and frame selection vector files
               when a previous frame selection vector file is missing and allowed
               for 1D frame selection vectors.
               
[SS] 08/08/23: Method of masking data frame changed to improve computation speed.    

[SS] 17/08/23: Added handling for multiple science cube inputs 

[SS] 22/08/23: Check for duplicate reductions of the same observation removes 
               any cubes with fewer frames before selecting most recent reduction.
               
[SS] 27/02/24: Frame rotation and timestamp inputs added + asm query

[SS] 09/03/24: File name handling moved to separate script

[SS] 08/04/24: Data read in with RetryHandler when process is carried out on SUMMER
"""

import argparse
import warnings
import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy.utils.exceptions import AstropyWarning
from skimage.draw import disk
import rdi_ref_table as rtab
import file_sorter
import query_asm

from file_sorter import read_file
#from utils.retry_handler import RetryHandler

warnings.simplefilter('ignore', category=AstropyWarning)
D_TEL = 8.2 #m
AO_WID = 20 #px; width of annulus for measuring ao correction radius correlation

"""
---------------------------------- FUNCTIONS ----------------------------------
"""
## reads in and returns science cube header if only 1 science target,
## otherwise returns a new header with copied keywords that have the same 
## value for all science cubes
def init_header(sci_path, ncube_sci):        
    if ncube_sci == 1:
        science_hdr = read_file(fits.getheader, sci_path)
        science_hdr['PATH_TAR'] = sci_path
        nx = science_hdr['NAXIS1']
    else:
        hdr_tmp = read_file(fits.getheader, sci_path)
        nx = hdr_tmp['NAXIS1']
        
        science_hdr = fits.Header()
        science_hdr['PIXSCAL'] = hdr_tmp['PIXSCAL']
        science_hdr['ESO INS COMB ICOR'] = hdr_tmp['ESO INS COMB ICOR']
        science_hdr['ESO INS COMB IFLT'] = hdr_tmp['ESO INS COMB IFLT']
    science_hdr['NB_SCI_CUBES'] = ncube_sci
    
    return science_hdr, nx
       
        
## add path, number of frames, and start frame index of each reference cube to the header
def complete_header(science_hdr, paths, nframe_ref):
    science_hdr.set('', '----------------')
    ncube_ref = len(paths)
    ind = 0
    for i in range(ncube_ref):
        nb_str = '{0:06d}'.format(i)
        ##TODO: change to different identifier
        science_hdr['RN'+nb_str] = paths[i]
        science_hdr['RF'+nb_str] = nframe_ref[i]
        science_hdr['RS'+nb_str] = ind
        ind += nframe_ref[i]
        
    science_hdr['COMMENT'] = '[RN]= reference cube path'
    science_hdr['COMMENT'] = '[RF]= no. frames in reference cube'
    science_hdr['COMMENT'] = '[RS]= start index of reference cube in matrix'


## create boolean mask for annulus within which correlation is calculated
def create_mask(size, inner_radius, outer_radius):
    cxy=(size//2,size//2)
    mask_in=disk(cxy,inner_radius,shape=(size,size))
    mask_out=disk(cxy,outer_radius,shape=(size,size))
    mask=np.full((size,size),False)
    mask[mask_out]=True
    mask[mask_in]=False

    return mask


## print science cube info + save info to header
def print_cube_info(sci_path, ncube_sci, isci, shape, nframes, iframe_0, science_hdr):
    hdr_tmp = read_file(fits.getheader, sci_path)
    if isci == 0:
        print('\n>> ESO INS COMB ICOR:', hdr_tmp['ESO INS COMB ICOR'])
        print('>> ESO INS COMB IFLT:', hdr_tmp['ESO INS COMB IFLT'])  
    
    print('\n------ Science cube {0:03d}/{1:03d} ------'.format(isci+1,ncube_sci))
    print('>> OBJECT:', hdr_tmp['OBJECT'])
    print('>> DATE-OBS:', hdr_tmp['DATE-OBS'])
    print('>> EXPTIME:', hdr_tmp['EXPTIME'])
    print('> science_cube.shape =', shape)
    if nframes != shape[1]:
        print('> no. frames after applying frame selection vector =', nframes)
    else:
        print('> science cube does not have a frame selection vector')
        
    if science_hdr is not None:
        nb_str = '{0:03d}'.format(isci)
        science_hdr['OBJ_'+nb_str] = hdr_tmp['OBJECT']
        science_hdr['PATH_'+nb_str] = sci_path
        science_hdr['FRMS_'+nb_str] = nframes
        science_hdr['FRM0_'+nb_str] = iframe_0
        science_hdr['OBS_'+nb_str] = hdr_tmp['DATE-OBS']
        science_hdr['DATE_'+nb_str] = hdr_tmp['DATE']
        science_hdr['REF_POSN_'+nb_str] = 0 #placeholder
        if isci == 0:
            science_hdr.comments['FRMS_'+nb_str] = 'no. frames in science cube'
            science_hdr.comments['FRM0_'+nb_str] = 'start index of science cube in matrix'
            science_hdr.comments['DATE_'+nb_str] = 'convert cube file created on this date'


#apply mask to frame and subtract mean    
def format_frame(frame, mask):
    masked_frame = frame[mask]
    return masked_frame-np.nanmean(masked_frame)


## get indices of data cube frames to keep from frame selection vector if file exists,
## otherwise all frames are used    
def selection_vector(frame_path, nframes_0):
    if frame_path != 'na':
        select_frames = read_file(fits.getdata, frame_path)
        if len(select_frames.shape)>1: 
            select_frames = select_frames[:,0]
        select_frames = np.where(select_frames==1.)[0]
    else:
        select_frames = np.arange(0,nframes_0)
        
    return select_frames
#%%
"""
---------------------------------- MAIN CODE ----------------------------------
"""
parser = argparse.ArgumentParser(description='Builds a marix containing the Pearson correlation coefficient calculated between each science and reference frame within a specified annulus.')
parser.add_argument('sof', help='File name of the sof file',type=str)
parser.add_argument('--inner_radius',help='Inner radius of correlation annulus.', type=int, default=10)
parser.add_argument('--outer_radius',help='Outer radius of correlation annulus.', type=int, default=50)
parser.add_argument('--wl_channels', help='Wavelength channel to use. 0=Channel 1, 1=Channel 2, 2=Channels 1 and 2', type=int, choices=[0,1,2], default=0)
parser.add_argument('--use_science', action='store_true', help='Calculate correlation of science cube with itself.')
parser.add_argument('--save_table', action='store_true', help='Save a table and csv containing observational conditions and stellar parameters of the reference cubes taken from the cube fits header.')
parser.add_argument('--ao_corr', action='store_true', help='Caclulate additional correlation within a 15px annulus at the AO correction radius.')

args = parser.parse_args()

## sof file path
sofname=args.sof

## --use_science flag [True||False]
USE_SCI = args.use_science

## --save_table flag [True||False]
SAVE_TABLE = args.save_table

## --wl_channels [0||1||2]
dict_conversion_wl_channels = {0 : [0], 1 : [1], 2 : [0,1]}
wl_channels = dict_conversion_wl_channels[args.wl_channels]
nb_wl = len(wl_channels)

## --inner/outer radius for comparison region [Z]
inner_radius = args.inner_radius
outer_radius = args.outer_radius

if outer_radius<=10:
    outer_radius=10
    print('Warning: outer radius too small. Setting to 10')

if outer_radius <= inner_radius:
    print('Warning: outer_radius <= inner_radius. Inner radius set to {0:d}'.format(outer_radius-1))

AO_CORR = args.ao_corr
#%%
                        #data type in sof file          #minimum no.files
data_names = {'cube':('IRD_SCIENCE_REDUCED_MASTER_CUBE', 1+USE_SCI),
              'lambda':('IRD_SCIENCE_LAMBDA_INFO', 1),
              'frame':('IRD_FRAME_SELECTION_VECTOR', 0),
              'parang':('IRD_SCIENCE_PARA_ROTATION_CUBE', 1+USE_SCI),
              'time':('IRD_TIMESTAMP', 1+USE_SCI)}

science_paths, ref_paths, skip_cube, lambda_path, star_data, columns = file_sorter.get_paths(sofname, data_names, USE_SCI)

ncube_ref = len(ref_paths['cube'])
if ncube_ref < 1:
    raise Exception('No eligible reference targets')

ncube_sci = len(science_paths['cube'])
#if ncube_sci==1 and not USE_SCI:
#    ncubes = ncube_ref+1
#else:
#    ncubes = ncube_ref
ncubes = len(set(ref_paths['cube']+science_paths['cube'])) ##no. unique file names
print('\n> Library contains', ncube_ref, 'reference star(s) and', ncube_sci, 'science target(s).')
if USE_SCI:
    print('> Science targets(s) included in reference library.')
science_hdr, nx = init_header(science_paths['cube'][0], ncube_sci)
pxscale = science_hdr['PIXSCAL']

wl_lambda = read_file(fits.getdata, lambda_path)
wl_to_ao_radius = lambda wl: int(u.rad.to(u.mas,20*wl*1e-6/D_TEL)/pxscale) ## 20 lambda/D in px = AO radius 
ao_radius = np.zeros(nb_wl, dtype=int)
for i,wl in enumerate(wl_channels):
    ao_radius[i] = wl_to_ao_radius(wl_lambda[wl])
if AO_CORR:    
    crop_size = 2*max(outer_radius, max(ao_radius)+AO_WID)+1
else:
    crop_size = 2*outer_radius+1

mask = create_mask(crop_size, inner_radius, outer_radius)
if AO_CORR: 
    print("> Calculating additional correlation for AO ring.")   
    ao_masks = np.empty(((nb_wl,)+mask.shape), dtype=bool)
    for i,r in enumerate(ao_radius):
        ao_masks[i] = create_mask(crop_size, r, r+AO_WID)

if (nx+crop_size)%2:
    nx+=1
border_l = (nx - crop_size)//2
border_r = (nx + crop_size)//2       

science_frames = []
sci_frame_select = []
sci_cube_nframe = np.zeros(ncube_sci, dtype=int)
sci_range = np.zeros(ncube_sci+1, dtype=int)

for isci, (sci_path, sci_frame_path) in enumerate(zip(science_paths['cube'], science_paths['frame'])):
    sci_cube = read_file(fits.getdata, sci_path)
    select_frames = selection_vector(sci_frame_path, sci_cube.shape[1])
    
    ## storing selecion vector, number of frames, and start + end frame index for each science cube
    sci_cube_nframe[isci] = len(select_frames)
    sci_range[isci+1] = sci_range[isci] + len(select_frames)
    sci_frame_select.append(select_frames)
    
    print_cube_info(sci_path, ncube_sci, isci, sci_cube.shape, len(select_frames), 
                    sci_range[isci], {0:science_hdr,1:None}[ncube_sci==1])
    
    science_frames.append(sci_cube[:,select_frames, border_l:border_r, border_l:border_r])
    del sci_cube

science_frames = np.concatenate(science_frames, axis=1)
sci_nframe = science_frames.shape[1]

corr_mat = []
ref_frame_select = []
ref_cube_nframe = np.zeros(ncube_ref, dtype=int)
if AO_CORR: corr_mat_ao = []

if SAVE_TABLE:
    target_info = rtab.init_info_dict(ncubes, columns)
sci_index = np.zeros(ncube_sci, dtype=int)  

for refi, (path, frame_path, parang_path, time_path) in enumerate(zip(ref_paths['cube'], ref_paths['frame'], 
                                                                      ref_paths['parang'], ref_paths['time'])):
    if path in science_paths['cube']: ## ref cube is a science cube --> frames already formatted
        isci = science_paths['cube'].index(path) ## index within the science targets list
        sci_index[isci] = refi ## index within the reference targets list
        
        select_frames = sci_frame_select[isci]
        ref_nframe = sci_cube_nframe[isci]
        i_0, i_1 = sci_range[isci:isci+2]
        reference_frames = science_frames[:,i_0:i_1]
        
    else: ## read in ref cube and format frames
        reference_frames = read_file(fits.getdata, path)
        select_frames = selection_vector(frame_path, reference_frames.shape[1])
        ref_nframe = len(select_frames)
        
        reference_frames = reference_frames[:,select_frames, border_l:border_r, border_l:border_r]
        
    ref_cube_nframe[refi] = ref_nframe
    
    timestamp = read_file(fits.getdata, time_path) + read_file(fits.getheader, time_path)['SUBTRACT']
    derot_ang = read_file(fits.getdata, parang_path)
    
    frms = np.zeros((4, ref_nframe))
    frms[0] = refi
    frms[1] = select_frames
    frms[2] = timestamp[select_frames]
    frms[3] = derot_ang[select_frames]
    
    ref_frame_select.append(frms)

    res = np.zeros((nb_wl, sci_nframe, ref_nframe))
    if AO_CORR: res_ao = np.zeros_like(res)
    ## calculating correlation
    for isci in range(ncube_sci):
        i_0, i_1 = sci_range[[isci,isci+1]]
        if path in skip_cube[isci]:
            res[:,i_0:i_1] = np.nan
            if AO_CORR: res_ao[:,i_0:i_1] = np.nan
            continue ## to next science target
        
        for wli,wl in enumerate(wl_channels):
            for i in range(i_0,i_1):
                sci_frame = format_frame(science_frames[wli,i],mask) 
                if AO_CORR: sci_frame_ao = format_frame(science_frames[wli,i], ao_masks[wli]) 
                for j in range(ref_nframe):
                    ref_frame = format_frame(reference_frames[wli,j],mask)
                    res[wli,i,j] = np.corrcoef(sci_frame, ref_frame)[0,1]
                    if AO_CORR:
                        ref_frame_ao = format_frame(reference_frames[wli,j], ao_masks[wli])
                        res_ao[wli,i,j] = np.corrcoef(sci_frame_ao, ref_frame_ao)[0,1]                        
    corr_mat.append(res)
    if AO_CORR: corr_mat_ao.append(res_ao)
    if SAVE_TABLE: 
        rtab.add_target_info(target_info, columns, refi, ref_paths['simbad_id'][refi], read_file(fits.getheader, path))   

if ncubes > ncube_ref: ##one or more science targets not used in ref cube
    count = refi + 1
    for i,path in enumerate(science_paths['cube']):
        if path in ref_paths['cube']:
            continue
        sci_index[i] = count
        
        timestamp = read_file(fits.getdata, science_paths['time'][i]) + read_file(fits.getheader, science_paths['time'][i])['SUBTRACT']
        derot_ang = read_file(fits.getdata, science_paths['parang'][i])
        
        frms = np.zeros((4, sci_cube_nframe[i]))
        frms[0] = sci_index[i]
        frms[1] = sci_frame_select[i]
        frms[2] = timestamp[sci_frame_select[i]]
        frms[3] = derot_ang[sci_frame_select[i]]
        
        ref_frame_select.append(frms)
        
        ref_paths['cube'].append('science {0:03d}'.format(i))
        ref_cube_nframe = np.append(ref_cube_nframe, sci_cube_nframe[i])
        if SAVE_TABLE: 
            rtab.add_target_info(target_info, columns, count, science_paths['simbad_id'][i], read_file(fits.getheader, path))
        count+=1
        
ref_frame_select=np.concatenate(ref_frame_select,axis=1)
corr_mat=np.concatenate(corr_mat, axis=-1)
if AO_CORR: corr_mat_ao=np.concatenate(corr_mat_ao, axis=-1)

print('\n> Correlation calculated between {0:d} science frames and {1:d} reference frames.'.format(sci_nframe, ref_cube_nframe.sum()))  

## complete header + save files
for i,refi in enumerate(sci_index):
    science_hdr['REF_POSN_{0:03d}'.format(i)] = refi
science_hdr['USE_SCI'] = (USE_SCI, 'science frame self correlation included in matrix')
science_hdr['INNER_R'] = inner_radius
science_hdr['OUTER_R'] = outer_radius
science_hdr['WL_CHOSE'] = args.wl_channels
for i,wl in enumerate(wl_channels):
    science_hdr['WL{0:d}_AO_R'.format(wl)] = ao_radius[i]
science_hdr['NB_REF_CUBES'] = ncube_ref

complete_header(science_hdr, ref_paths['cube'], ref_cube_nframe)

print('..Saving correlation matrix..')
hdu = fits.PrimaryHDU(data=corr_mat, header=science_hdr)
hdu.writeto('pcc_matrix.fits')

if AO_CORR:
    change_type = lambda x: int(x[0]) if x.size==1 else str(x)
    science_hdr['INNER_R'] = change_type(ao_radius)
    science_hdr['OUTER_R'] = change_type(ao_radius + AO_WID)
    
    print('..Saving AO ring correlation matrix..')
    hdu_ao = fits.PrimaryHDU(data=corr_mat_ao, header=science_hdr)
    hdu_ao.writeto('pcc_matrix_ao_ring.fits')

## frames of ref cubes after frame selection vector is applied
hdu_frame = fits.PrimaryHDU(data=ref_frame_select, header=science_hdr)
hdu_frame.writeto('corr_frame_selection.fits')

if SAVE_TABLE:
    ref_table = rtab.make_ref_table(target_info, columns, star_data)
    print('..Querying ASM..')
    df, ecmwf_wind = query_asm.timestamp_query(ref_table, ref_frame_select, ncubes)
    
    print('..Saving reference frame ASM data..')
    df.to_csv("ref_frame_asm_data.csv")
    
    for i,(col,unit) in enumerate(zip(['Wind_speed_ecmwf','Wind_dir_ecmwf'],['m/s','deg'])):
        ref_table[col] = ecmwf_wind[:,i]
        ref_table[col].unit = unit  
    
    print('..Saving reference target table..')
    table_hdu = fits.BinTableHDU(ref_table, header=science_hdr)
    table_hdu.writeto('target_info_table.fits')