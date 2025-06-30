'Query ASM for reference cube frames using timestamp.'

import warnings
import numpy as np
import pandas as pd
import subprocess
import urllib.parse as urlp

from astropy.time import Time
from astropy.utils.exceptions import AstropyWarning
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.simplefilter('ignore', category=AstropyWarning)
#%%
PARAM_STR = {'csv': {'seeing': 'dimm_seeing', ## filename of csv file for saving wget output
                     'tau0': 'massdimm_tau0',
                     'ecmwf': 'ecmwf_200mbar_wind',
                     'meteo': 'meteo_30m_wind',
                     'hist': 'hist_ambi'},
             
             'url': {'seeing': 'dimm/fwhm', ## instrument/variable string for cobrex asm query
                     'tau0': 'mass/tau0', #massdimm
                     'ecmwf': 'ecmwf_era5_reanalysis/wind*_200',
                     'meteo': 'meteo/wind_*1',
                     'hist': 'historical_ambient'},
             
             'col': {'fwhm': 'seeing', ## column name of asm wget convert to column name in final data csv
                     'tau': 'tau0',
                     'tau0': 'tau0',
                     'windPhi_200': 'wind_dir_200mbar',
                     'windSpeed_200': 'wind_speed_200mbar',
                     'wind_dir1': 'wind_dir_30m',
                     'wind_speed1': 'wind_speed_30m'},
             
             'var': {'seeing': ['seeing'], ## final csv column names for each param 
                     'tau0': ['tau0'],
                     'meteo': ['wind_speed_30m','wind_dir_30m'],
                     'ecmwf': ['wind_speed_200mbar','wind_dir_200mbar'],
                     'hist': ['seeing','tau0']},
             
             'table': {'seeing': ['FWHM_mean_DIMM',['FWHM_tel_ambi_start','FWHM_tel_ambi_end']], ## column name in ref data table
                       'tau0': ['Tau0_mean_DIMM','Tau0_tel_ambi'],
                       'wind_speed_30m': 'Wind_speed_tel_ambi',
                       'wind_dir_30m': 'Wind_dir_tel_ambi'}}

MASSDIMM_START = Time('2016-04-05T23:49:00')
COL_NAMES = ['seeing', 'tau0', 'wind_dir_200mbar', 'wind_dir_200mbar_onframe', 'wind_speed_200mbar', 'wind_dir_30m', 'wind_dir_30m_onframe', 'wind_speed_30m']

TABLE_PATH = 'target_parameter_table.fits'

def wget_asm(date, p, end_date=None):
    filename = '{0:s}_data.csv'.format(PARAM_STR['csv'][p])
    
    url = "http://cobrex-dc.osug.fr:8080/asm-server/api/asm/csv/{0:s}/{1:s}".format(PARAM_STR['url'][p], date.value)
    if end_date is not None: url+='..{0:s}'.format(end_date.value)
    url_esc = urlp.quote_plus(url, safe=':/')
    
    request_asm_str = ['wget', '-O',filename, url_esc]
    output,error = subprocess.Popen(request_asm_str, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()
    df = pd.read_csv(filename, sep=';', skiprows=0, skipfooter=0, parse_dates=True, quoting=3, index_col=['name'])
    
    return df

## no asm data available for ecmwf for observation time so interpolate asm data within 12hrs
def interp_wind(epoch, data, framei, parang):
    try:
        df = wget_asm(epoch-0.25, 'ecmwf', epoch+0.25)
    except pd.errors.EmptyDataError:
        return
    
    time_list = Time(list(df.loc['windSpeed_200','dateTime']))
    time_list.format='isot'
    
    interp_func_speed = interp1d(time_list.mjd, np.array((df.loc['windSpeed_200','value'])), 
                                 kind='linear', bounds_error=False, fill_value="extrapolate")
    unwrap_winddir = np.rad2deg(np.unwrap(np.deg2rad(np.array((df.loc['windPhi_200','value'])))))
    interp_func_dir = interp1d(time_list.mjd, unwrap_winddir, kind='linear',
                               bounds_error=False, fill_value="extrapolate")
    
    data['wind_speed_200mbar'][framei] = interp_func_speed(epoch.mjd)
    data['wind_dir_200mbar'][framei] = interp_func_dir(epoch.mjd)
    format_dir(data, 'wind_dir_200mbar', framei, parang)

## no asm data available so using cube value from header (table) if available
def from_table(param, data, framei, t_row, parang):
    for col in param:
        t_colname = PARAM_STR['table'][col]
        if col in ['seeing','tau0']:
            v = t_row[t_colname[0]]
            if not np.isfinite(v):
                if col == 'seeing':
                    v = np.nanmean((t_row[t_colname[1][0]],t_row[t_colname[1][1]]))
                elif col == 'tau0':
                    v = t_row[t_colname[1]]
        else:
            v = t_row[t_colname]
        
        if not np.isfinite(v):
            return
        else:
            data[col][framei] = v
            if 'dir' in col: format_dir(data, col, framei, parang)
            
    
def get_values(epoch, times, framei, data, t_row, parang):
    epoch_start, epoch_end = Time(epoch)
    if epoch_start > MASSDIMM_START:
        p_list = ['seeing','tau0','ecmwf','meteo']
    else:
        p_list = ['hist','ecmwf','meteo']
    
    for p in p_list:
        recalc_timei=False
        try:
            df = wget_asm(epoch_start, p, epoch_end)
        except pd.errors.EmptyDataError:
            if p != 'ecmwf':
                from_table(PARAM_STR['var'][p], data, framei, t_row, parang)
            else:
                interp_wind(epoch_start+(epoch_end-epoch_start)/2, data, framei, parang)
            continue
        
        index = list(set(df.index))
        ntimes = df.nunique()['dateTime']
        if ntimes == 1:
            timei = np.zeros_like(times, dtype=int)
        elif ntimes * len(index) != df.shape[0]:
            recalc_timei = True
        else:
            datetime = Time(list(df.loc[index[0],'dateTime']),format='isot').mjd
            timei = match_time(datetime, times)
            
        for name in index:
            if recalc_timei:
                if type(df.loc[name,'dateTime'])==str:
                    ntimes=1
                    timei = np.zeros_like(times, dtype=int)
                else:
                    ntimes=df.nunique()['dateTime']
                    datetime = Time(list(df.loc[name,'dateTime']),format='isot').mjd
                    timei = match_time(datetime, times)
                
            col = PARAM_STR['col'][name]
            if ntimes==1:
                data_tmp = np.array([df.loc[name,'value']])
            else:
                data_tmp = df.loc[name,'value'].to_numpy()
              
            data[col][framei] = data_tmp[timei]
            if 'dir' in col: format_dir(data, col, framei, parang)

## add frame rotation to wind direction to get an on frame angle
def format_dir(data, col_tmp, framei, parang):
    data[col_tmp+'_onframe'][framei] =  data[col_tmp][framei] + parang
    for col in [col_tmp, col_tmp+'_onframe']:
        if any(data[col][framei] < 0) or any(data[col][framei] > 360):
            for i in framei:
                while data[col][i] < 0: data[col][i]+=360
                while data[col][i] > 360: data[col][i]-=360
        
## indices of asm datetime query that are the closest match to the frame timestamps
def match_time(datetime, times):
    index=np.zeros(len(times),dtype=int)
    for i,t in enumerate(times):
        index[i] = find_nearest(datetime,t)
    return index
  
    
def find_nearest(array, value):
    return (np.abs(array-value)).argmin()


def timestamp_query(ref_table, frame_vect, ncubes):    
    timestamps = frame_vect[2]
    parang = frame_vect[3]
    
    data = {x:np.full(len(timestamps),np.nan) for x in COL_NAMES}
    data['timestamp_mjd'] = timestamps
    
    ecmwf_wind = np.full((ncubes,2),np.nan)
    for i in range(ncubes):
        fi=np.where(frame_vect[0]==i)[0]
        epoch_range = Time(timestamps[[fi[0],fi[-1]]],format='mjd').isot
        get_values(epoch_range, timestamps[fi], fi, data, ref_table[i], parang[fi])
        
        ecmwf_wind[i,0] = np.nanmean(data['wind_speed_200mbar'][fi]) 
        ecmwf_wind[i,1] = np.nanmean(data['wind_dir_200mbar'][fi])
        
    return pd.DataFrame(data=data), ecmwf_wind