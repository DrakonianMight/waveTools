# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 20:42:30 2020

This program is free for distribution and use as per the MIT licence.

This script contains a series of functions for ingesting datawell file format
data into python. The functions are designed to read in the data and return a directional spectrum.


Author: Leo Peach
"""

import pandas as pd
import numpy as np
import xarray as xr


def readDisplacement(filepath, siteid = None):
    """
    Reads in datawell displacement csv files and returns a dataframe.

    Parameters
    ----------
    filepath : str
        filepath to the csv file of the displacement data
    siteid : str or int, optional
        An optional site identification field. The default is None.

    Returns
    -------
    dipl : dataframe
        a dataframe of the displacement data

    """
    
    dipl = pd.read_csv(filepath, sep = '\t', header = None)
    dipl.columns = ['Time','status','heave','north','west']
    
    dipl.index = pd.to_datetime(dipl.Time, unit = 's')
    dipl.index.name = 'datetime'
    
    # if we want to add a site identifier we can
    if siteid is not None:
        dipl['siteid'] = siteid
    
    return dipl

def datawellFreqBins1d():
    """Datawell Mk4 frequency bins
    

    Returns
    -------
    list
        list of 1d frequency bins

    """
    
    low = []
    start = 0.025
    low.append(start)
    for i in range(45):
        start+=0.005
        low.append(np.round(start,3))
    mid = []
    start = 0.26
    mid.append(start)
    for i in range((78-46)):
        start+=0.01
        mid.append(np.round(start,3))
    high = []
    start = 0.6
    high.append(start)
    for i in range((100-80)):
        start+=0.02
        high.append(np.round(start,3))
    
    return low+mid+high

def readSpectrum1D_f20(filepath, siteid = None):
    """
    readDatawell 1d spectrum file (f20) csv
    

    Parameters
    ----------
    filepath : str
        filepath to the csv file of the spectrum data
    siteid : str or int, optional
        An optional site identification field. The default is None.

    Returns
    -------
    spectrum: dataframe
        a dataframe of the wave spectrum (1d)

    """
    
    spectrum = pd.read_csv(filepath, sep = '\t', header = None)
    
    spectrum.columns = ['timestamp','datestamp','segments']+datawellFreqBins1d()
    spectrum.index = pd.to_datetime(spectrum.timestamp, unit = 's')
    spectrum.index.name = 'datetime'
    
    # if we want to add a site identifier we can
    if siteid is not None:
        spectrum['siteid'] = siteid
        
    return spectrum



def readPrimDirSpectrum(filepath, siteid = None):
    """
    readDatawell Primary Directional Spectrum File (f21)
    The Direction fromk is the mean wave direction in bin k. This is the direction the waves are travelling from.  
    The Spreadk is the directional spread about the mean wave direction. 
    

    Parameters
    ----------
    filepath : str
        filepath to the csv file of spectrum data
    siteid : str or int, optional
        An optional site identification field. The default is None.

    Returns
    -------
    spectrum: dataframe
        a dataframe of the Primary Directional Spectrum
        
    Notes
    -----
    More details are available in the datawell manual:
    https://www.datawell.nl/Portals/0/Documents/Manuals/datawell_specification_csv_file_formats_s-02-v1-6-0.pdf

    """
    
    def getColumnNames():
        #build the column names
        spreadNames = ['spr'+str(i) for i in range(100)]
        dirNames = ['dir'+str(i) for i in range(100)]
        return ['timestamp','daystamp','segments']+dirNames+spreadNames
    
    
    Primdirspec = pd.read_csv(filepath, sep = '\t', header = None)
    
    Primdirspec.columns = getColumnNames()
    Primdirspec.index = pd.to_datetime(Primdirspec.timestamp, unit = 's')
    Primdirspec.index.name = 'datetime'
    
    # if we want to add a site identifier we can
    if siteid is not None:
        Primdirspec['siteid'] = siteid
        
    return Primdirspec


def readSecDirSpectrum(filepath, siteid = None):
    """
    readDatawell Secondary Directional Spectrum File (f28)
    The secondary directional spectrum message contains the second-harmonic coefficients and the checkfactor K. 
    
    The (m2)k is the centred cosine Fourier coefficient in bin k.  
    The (n2)k is the centred sine Fourier coefficient in bin k. 

    Parameters
    ----------
    filepath : str
        filepath to the csv file of spectrum data
    siteid : str or int, optional
        An optional site identification field. The default is None.

    Returns
    -------
    spectrum: dataframe
        a dataframe of the Secondary Directional Spectrum Message
    
    Notes
    -----
    More details are available in the datawell manual:
    https://www.datawell.nl/Portals/0/Documents/Manuals/datawell_specification_csv_file_formats_s-02-v1-6-0.pdf

    """
    
    def getColumnNames():
        # build columns names
        cosine = ['cosine_'+str(i) for i in range(100)]
        sine = ['sine_'+str(i) for i in range(100)]
        checkF = ['check_'+str(i) for i in range(100)]
        return ['timestamp','daystamp','segments']+cosine+sine+checkF
    
    Secdirspec = pd.read_csv(filepath, sep = '\t', header = None)

    Secdirspec.columns = getColumnNames()
    Secdirspec.index = pd.to_datetime(Secdirspec.timestamp, unit = 's')
    Secdirspec.index.name = 'datetime'
    
    # if we want to add a site identifier we can
    if siteid is not None:
        Secdirspec['siteid'] = siteid
        
    return Secdirspec
    
    
def readSpecWaveParams(filepath, siteid = None):
    """
    readDatawell CSV file spectrally derived wave parameters {f25}
    
    Parameters
    ----------
    filepath : str
        filepath to the csv file of spectrum data
    siteid : str or int, optional
        An optional site identification field. The default is None.

    Returns
    -------
    waveParams : dataframe
        Dataframe of the wave parameters.

    """
    
    waveParams = pd.read_csv(filepath, sep = '\t', header = None)
    waveParams.columns = ['timestamp','daystamp','segments','Hs','Ti'
                            ,'Te','T1','Tz','T3','Tc','Rp','Tp','Smax'
                            ,'dirP(rad)','dirP_spread(rad)']
    waveParams.index = pd.to_datetime(waveParams.timestamp, unit = 's')
    waveParams.index.name = 'datetime'
    
    # if we want to add a site identifier we can
    if siteid is not None:
        waveParams['siteid'] = siteid
    
    return waveParams

    
def calculateDirectionalSpectrum(spectrum1D, primDirSpec, secDirSpec, Stats, direction_step=10):
    """
    Calculate the directional wave spectrum from 1D spectrum, primary and secondary directional spectra.

    Parameters
    ----------
    spectrum1D : dataframe
        Dataframe of the 1D wave spectrum.
    primDirSpec : dataframe
        Dataframe of the primary directional spectrum.
    secDirSpec : dataframe
        Dataframe of the secondary directional spectrum.
    direction_step : int, optional
        Angular resolution for directions (default is 5 degrees).

    Returns
    -------
    directional_spectrum_ds : xarray.Dataset
        Dataset of the directional wave spectrum.
        
    Notes:
    Currently overcooks wave heights>>> too much energy.....
    """
    
    from wavespectra.specarray import SpecArray
    from wavespectra.specdataset import SpecDataset
    
    directions_deg = np.arange(0, 360, direction_step)  # Directions from 0 to 350 degrees in 10-degree increments
    directions_rad = np.deg2rad(directions_deg) 
    timestamps = primDirSpec.sort_index().index
    frequencies = datawellFreqBins1d()
    

    # Initialize an empty array to store the directional wave spectra
    spectrum_data = np.zeros((len(timestamps), len(frequencies), len(directions_deg)))

    for t_idx, timestamp in enumerate(timestamps):
        primary_data = primDirSpec.loc[timestamp]
        secondary_data = secDirSpec.loc[timestamp]

        S_f = spectrum1D.loc[timestamp][frequencies].values /100  # 1D energy spectrum
        # Scale the 1D spectrum to match the Smax for the current timestamp from the datawell data
        #Smax = Stats.loc[timestamp]['Smax']
        #max_1D_value = S_f.max()
        #if max_1D_value > 0:
        #    scaling_factor_1D = Smax / max_1D_value
        #    S_f *= scaling_factor_1D
        #    S_f = S_f /1000 #convert to m^2/Hz

        for f_idx, freq in enumerate(frequencies):
            #aka a1 and b1
            mean_dir = primary_data[f'dir{f_idx}']
            #mean_dir_to = (mean_dir + np.pi) % (2 * np.pi)
            spread = primary_data[f'spr{f_idx}']
            
            a2 = secondary_data[f'cosine_{f_idx}']  # Second harmonic cosine coefficient
            b2 = secondary_data[f'sine_{f_idx}']  # Second harmonic sine coefficient

            # Compute the directional spreading function
            D_theta = (1 + spread * np.cos(directions_rad - mean_dir) +
                       a2 * np.cos(2 * (directions_rad - mean_dir)) + 
                       b2 * np.sin(2 * (directions_rad - mean_dir))) ** 2
            
            # Normalize the spreading function
            normalization_factor = np.trapz(D_theta, directions_rad)
            if normalization_factor != 0:
                D_theta /= normalization_factor

            # Compute the directional wave spectrum
            E_theta = S_f[f_idx] * D_theta

            spectrum_data[t_idx, f_idx, :] = E_theta
            
    directional_spectrum = xr.DataArray(
        data=spectrum_data,
        coords={'time': timestamps.values,
                'freq': frequencies,
                'dir': directions_deg},
        dims=('time', 'freq', 'dir'),
        name='efth')
    
        # Scale the 2D spectrum to match Smax for each timestamp
    for t_idx, timestamp in enumerate(timestamps):
        max_2D_value = directional_spectrum.sel(time=timestamp).max().item()
        Smax = Stats.loc[timestamp]['Smax']
        if max_2D_value > 0:
            scaling_factor_2D = Smax / max_2D_value
            directional_spectrum.loc[dict(time=timestamp)] *= scaling_factor_2D
        
    directional_spectrum.attrs.update({
        'units': 'm^2/Hz/deg',
        'standard_name': 'sea_surface_wave_directional_variance_spectrum',
        'long_name': 'Directional Wave Spectrum'
    })
    #apply the scaling factor on exit
    return directional_spectrum /100
