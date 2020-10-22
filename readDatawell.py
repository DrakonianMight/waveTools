# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 20:42:30 2020

This program is free for distribution and use as per the MIT licence.

This script contains a series of functions for ingesting datawell file format
data into python.


Author: Leo Peach
"""

import pandas as pd
import numpy as np


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
    dipl.columns = ['timestamp','status','h','n','w']
    
    dipl.index = pd.to_datetime(dipl.timestamp, unit = 's')
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

    Parameters
    ----------
    filepath : str
        filepath to the csv file of spectrum data
    siteid : str or int, optional
        An optional site identification field. The default is None.

    Returns
    -------
    spectrum: dataframe
        a dataframe of the Secondary Directional Spectrum
    
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
        readDatawell CSV file spectrally derived wave parameters.

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
                              ,'Te','T1','Tz','T3','Tc','Tp','Tp','Smax'
                              ,'dirP(rad)','dirP_spread(rad)']
        waveParams.index = pd.to_datetime(waveParams.timestamp, unit = 's')
        waveParams.index.name = 'datetime'
        
        # if we want to add a site identifier we can
        if siteid is not None:
            waveParams['siteid'] = siteid
        
        return waveParams
    
    
    
    
    
    
    