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
    None.

    """
    
    spectrum = pd.read_csv(filepath, sep = '\t', header = None)
    
    spectrum.columns = ['timestamp','datestamp','segments']+datawellFreqBins1d()
    spectrum.index = pd.to_datetime(spectrum.timestamp, unit = 's')
    spectrum.index.name = 'datetime'
    
    # if we want to add a site identifier we can
    if siteid is not None:
        spectrum['siteid'] = siteid
        
    return spectrum
    