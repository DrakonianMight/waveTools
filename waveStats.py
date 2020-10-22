# -*- coding: utf-8 -*-
"""Wave Statistics Tools

A series of tools designed to perform wave analysis.
Uses a class object to wrap and peform analysis.

This is authored under MIT licence.

Author: Leo Peach

Started: 15-05-2020
Verson: 1.2

Changelog:

v0.1 - added displacements
v0.4 - added 1d displacements class object
v1.0 - added 2d spectra functionality
v1.2 - added filter tools
"""

import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits import mplot3d
from scipy.signal import butter, lfilter, filtfilt


class displacements_1d(object):
    """Non Directional Displacement data class. 
    
    A displacement data class with a series of useful methods.
    
    Parameters
    ----------
    dataframe : DataFrame
        A dateframe containing: seconds, heave, north, west
    
    Attributes
    ----------
    filepath : str
        A filepath to the data:time, heave, north, west
    array : ndarray
        (time, heave, north, west) all the displacement data 
    time : ndarray
        (time) all the times from the data
    heave : ndarray
        (heave) all the vertial heave displacements
    north : ndarray
        (north) all the north displacements
    west : ndarray
        (west) all the west displacements
    crossings : list , list
        (waveHeights, wavePeriods) from zero upcrossing analysis
    waveHeights : list
        a list of wave heights in metres
    wavePeriods : list
        a list of wave period in seconds
    spectra : ndarray
        (frequencies, power) using standard methods by default can
        be modified
    f : ndarrary
        frequencies from the spectral analysis
    power : ndarray
        power spectral density array from the spectral analysis

        
    Methods
    -------
    wave_stats : DataFrame
        a dataframe object containing wave statistics most commonly used
    """
    
    def __init__(self, dataframe=None):
        """Extract the time and 3 displacement fields
        """
        self.df = dataframe
        
        array = self.df.values
        self.array = array
        self.time = array[:, 0]
        self.heave = array[:, 1]
        self.crossings()
        self.f , self.power = self.spectra()
        
    def data(self):
        """A simple method of returning the data object
        """
        return self.array
        
    def summary(self):
        """prints a summary of the displacement data 
        """
        print("records in dataset: "+str(len(self.heave)))
    
    def sampling_frequency(self):
        """Retrieves the sampling frequency of the displacements.
        
        Parameters
        ----------
        sampling_frequency : ndarray
            time 2 - time1
        
        Returns
        -------
        float : sampling frequency in hz
        
        """
        #sampling frequecy in seconds
        sample_seconds = self.time[1] - self.time[0]
        
        sf = round(1/sample_seconds, 2)
        
        return sf
    
    def crossings(self):
        """Analyses the displacements to calculate the number of waves, periods and times
        
        Parameters
        ----------
        time : nparray
            time array in seconds
        heave : nparray
            heave array
        Returns
        -------
        waveHeights : list
            list wave heights
        wavePeriods : list
            list of wave periods
        times : list
            list of times
        
        """
        
        def pairwise(iterable):
            """
            Yields pairs of items, i.e.
            s -> (s0,s1), (s1,s2), (s2, s3), ...
            Adopted from http://stackoverflow.com/a/5764807
            """
            from itertools import tee
            a, b = tee(iterable)
            next(b, None)
            return zip(a, b)
        
        waveHeights = []
        wavePeriods = []
        times = []
        
        is_positive = self.heave > 0.001  #10mm grace allowed for the zero upcross
        crossings = ( ~is_positive[:-1] & is_positive[1:]).nonzero()[0]
        for waveStart, waveEnd in pairwise(crossings):
            waveDisplacements = self.array[waveStart:waveEnd,:]
            waveHeights.append(max(waveDisplacements[:,1]) - min(waveDisplacements[:,1]))
            wavePeriods.append((self.array[waveEnd, 0]) - (self.array[waveStart, 0]))
            times.append(self.array[waveStart,0])
            
        self.waveHeights = waveHeights
        self.wavePeriods = wavePeriods
            
        return waveHeights, wavePeriods
    
    def plot_waveheights(self):
        """plots ranked wave heights
        
        Returns
        -------
        fig : matplotlib figure obj
            Figure object
        ax : matplotlib axis obj
            Axis object
        """
        waveHeights = sorted(self.waveHeights, reverse=True)
        plt.clf()
        fig , ax = plt.subplots()
        ax.plot(waveHeights)
        plt.title('ranked wave heights')
        plt.ylabel('wave height (m)')
        plt.show()
        
        return fig, ax
        
    
    def hsig(self):
        """Calculates the zero upcrossing signficant wave height.
        
        The signficant wave height, the average wave height from the largest
        1/3 of waves in a given time series.
        
        Parameters
        ----------
        waveHeights : list
            a list containing the wave heights
        
        Returns
        -------
        Hsig : float
        
        """
        
        waveHeights = sorted(self.waveHeights, reverse=True)
        numberofWaves = len(waveHeights)
        HsWaveHeights = waveHeights[:numberofWaves//3]
        hsig = sum(HsWaveHeights) / float(len(HsWaveHeights))
    
        return round(hsig, 3)
    
    def tz(self):
        """The mean wave period, Tz, defined as the aver periods of waves from
        upcrossing analysis.
        
        Returns
        -------
        Tz : float
        
        """
        Tz = sum(self.wavePeriods) / len(self.wavePeriods)
        return round(Tz, 3)
    
    def hmax(self):
        """
        The maximum wave height, Hmax, represents the maximum value of the wave
        height measured over a given period of time.
        
        Returns
        -------
        hmax : float
        """
        return round(max(self.waveHeights), 3)
    
    def h10(self):
        """
        H10 is defined as the average of the largest 1/10 waves in a record. 
        The determination of H10 is based on the zero up-crossing method
           
        Returns
        -------
        h10 : float
        """
        waveHeights = sorted(self.waveHeights, reverse=True)
        numberofWaves = len(waveHeights)
        H10WaveHeights = self.waveHeights[:numberofWaves//10]
        H10 = sum(H10WaveHeights) / float(len(H10WaveHeights))
        return round(H10, 3)
    
    def ts(self):
        """
        The significant wave period, represented by T1/3 or Ts is defined as the
        average period of the largest 1/3 waves in a record
           
        Returns
        -------
        tz : float
        """
        sortedWavePeriods = [x[1] for x in sorted(zip(self.waveHeights,self.wavePeriods), 
                                              key=lambda x: x[0], reverse=True)]
        
        numberofWaves = len(sortedWavePeriods)
        TsWavePeriods = sortedWavePeriods[:numberofWaves//3]
        Ts = sum(TsWavePeriods) / float(len(TsWavePeriods))
        return round(Ts, 3)
        
    def hmean(self):
        """
        Hmean is defined as the average of the wave heights
        
        Returns
        -------
        hmean : float
        """
        Hmean = sum(self.waveHeights) / float(len(self.waveHeights))
        return round(Hmean, 3)

    def tmax(self):
        """
        The maximum wave period, Tmax, represents the maximum value of the
        discrete wave periods measured
        
        Returns
        -------
        tmax : float
        """
        Tmax = float(max(self.wavePeriods))
        return round(Tmax, 3)     
    
    def hrms(self):
        """
        The RMS of wave heights
        
        Returns
        -------
        hrms : float
        """
        numberofWaves = float(len(self.waveHeights))
        squaredWaveHeights = [x**2/numberofWaves for x in self.waveHeights]
        Hrms = sum(squaredWaveHeights) ** 0.5
        return round(Hrms, 3)
        
    def spectra(self, window = None, overlap = True):
        """Calculates a non-directional spectra using welch's method the default 
        uses and overlapping method with a Hann window of 17 segments.
        
        window length based on 4608 records in 30 mins -250 to make 28.4
        
        
        Parameters
        ----------
        heave : nparray
            nparray of the heave data
        sf : float
            sampling frequency of the data
        window : nparray
            window method to use in welch's method when adding a new window 
            the signal processing library for scipy should be used. e.g. 
            scipy.signal.hann
        overlap : Bool
            default value is True however if wanted this can be set to false 
            depending on the window type.
        
        
        Returns
        -------
        f : nparray
            a list of the different frequencies output from the welch's method
            fft calcuation
        Pxx_den : nparray
            a list of power spectral density values for each frequency.
            
        See Also
        --------
        * 'Scipy.Signal <https://docs.scipy.org/doc/scipy/reference/signal.html>'_
        """
        #setup welch method setup we use 17 segments (-250 records to match mk4)
        if self.sampling_frequency() == 2.56:
            windowlen = int(((len(self.heave)-250)/17)*2)
            if overlap == True:
                overlap = int(windowlen // 2)
    
            if window == None:
                window = signal.hann(windowlen)
            heave = self.heave[:-250]
                
        #setup bartletts method using 8 segments non overlapping
        if self.sampling_frequency() == 1.28:
            windowlen = 256
            #cut down displacements to 26.6 minutes
            heave = self.heave[:-262]
            
            #window function
            def datawell_tukey(n, alpha):
                '''
                Custom implementation of the Tukey window as done
                in PB's spreadsheet. Does not use (N-1) terms, just N
                '''
        
                window = np.ones(n)
                
                lower_check = np.divide(alpha*n, 2)
                middle_check = n*(1 - np.divide(alpha, 2)) - 1
                
                for i in range(n):
                    if i < lower_check:
                        window[i] = 0.5 * (1 + np.cos(np.pi * (np.divide(2*i, alpha*n) - 1)))
                    elif i > middle_check:
                        window[i] = 0.5 * (1 + np.cos(np.pi * (np.divide(2*(i + 1), alpha*n) - np.divide(2,alpha) + 1)))
                
                return window
            
            datawell_window = datawell_tukey(windowlen, 0.25)
            window = datawell_window
            
            overlap = None
            
        f, Pxx_den = signal.welch(heave, fs = self.sampling_frequency() ,
                                  window = window , noverlap = overlap)
        
        return f, Pxx_den
        
    def plot_spectra(self, semi = True):
        """Creates a spectrograph of the wave spectra.
        
        Parameters
        ----------
        semi : Bool
            plot style, default is semilogy
        
        Returns
        -------
        fig : matplotlib figure obj
            Figure object
        ax : matplotlib axis obj
            Axis object
        """
        f, Pxx_den = self.spectra()
        plt.clf()
        fig , ax = plt.subplots()
        
        ax.plot(f, Pxx_den)
        plt.title('spectrum')
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        
        return fig, ax
    
    def tp(self):
        """
        Peak wave period as defined as the frequency of the peak wave energy.
        converted to seconds.
        
        Returns
        -------
        tp : float
        """
        maxval = np.argmax(self.power)
        tp = 1/(self.f[maxval])
        return round(tp, 3)
        
    def hm0(self):
        """
        Signifcant wave height as calculated from the power spectrum zeorth moment.
        
        Returns
        -------
        hm0 : float
            meters
        """       
        #freq bin width
        freq_width = self.f[1] - self.f[0]
        #energy
        sum0 = np.sum(self.power)
        
        #calculate moment 0
        m0 = sum0 * freq_width
        
        hm0 = 4*np.sqrt(m0)
        
        return round(hm0 , 3)
    
    def t02(self):
        """
        Mean wave period as defined from the spectra.
        
        * uses a trapeziodal method to estimate the moments.
        
        Return
        ------
        t02 : float
            seconds
        """
        
        #calculate moment 0
        m0 = np.trapz(self.power * (self.f ** 0), self.f)
        #calculate moment 2
        m02 = np.trapz(self.power * (self.f ** 2), self.f)
        
        t02 = np.sqrt(m0 /m02)
        
        return round(t02, 3)
    
    def wave_stats(self):
        """
        Wave Statistics summary
        
        Returns
        -------
        wave stats : DataFrame
            a dataframe of the summary wave statistics
        
        """
        nameList = ['hsig','hmax','tz','tp','hm0','t02']
        sumList = [self.hsig(), self.hmax(), self.tz(), self.tp(), self.hm0(), 
                   self.t02()]
        summary = pd.DataFrame(sumList)
        summary.index = nameList
        
        return summary



class displacements(object):
    """Displacement data class. 
    
    A displacement data class with a series of useful methods.
    
    Parameters
    ----------
    dataframe : DataFrame
        A dateframe containing: seconds, heave, north, west
    
    Attributes
    ----------
    filepath : str
        A filepath to the data:time, heave, north, west
    array : ndarray
        (time, heave, north, west) all the displacement data 
    time : ndarray
        (time) all the times from the data
    heave : ndarray
        (heave) all the vertial heave displacements
    north : ndarray
        (north) all the north displacements
    west : ndarray
        (west) all the west displacements
    crossings : list , list
        (waveHeights, wavePeriods) from zero upcrossing analysis
    waveHeights : list
        a list of wave heights in metres
    wavePeriods : list
        a list of wave period in seconds
    spectra : ndarray
        (frequencies, power) calcuated using standard methods by default can
        be modified
    f : ndarrary
        frequencies as calcuated from the spectral analysis
    power : ndarray
        power spectral density array calculated from the spectral analysis
    dir_spectra : DataFrame
        a dataframe object containing the directional spectra
    dir_spectra_array : ndarray
        (127, 37) containing the directional spectra applied across 360 degrees
        
    Methods
    -------
    wave_stats : DataFrame
        a dataframe object containing wave statistics most commonly used
    """

    def __init__(self, dataframe=None):
        """Extract the time and 3 displacement fields
        """
        self.df = dataframe
        
        array = self.df.values
        self.array = array
        self.time = array[:, 0]
        self.heave = array[:, 1]
        self.north = array[:, 2]
        self.west = array[:, 3]
        self.crossings()
        self.f , self.power = self.spectra()
        self.dir_spectra, self.dir_spectra_array = self.directional_spectra()
        
    def data(self):
        """A simple method of returning the data object
        """
        return self.array

    def plot_heave(self):
        """Plots the heave data in multiple subplots for inspection
        

        Returns
        -------
        fig : matplotlib figure obj
            Figure object
        ax : matplotlib axis obj
            Axis object

        """
        fig, (ax1,ax2, ax3,ax4) = plt.subplots(4, 1, figsize = (20,10))
        
        div = len(self.df.heave)//4
        
        self.df.iloc[0:div].heave.plot(ax = ax1)
        div2 = div + div
        self.df.iloc[div:div2].heave.plot(ax = ax2)
        div3 = div2 + div
        self.df.iloc[div2:div3].heave.plot(ax = ax3)
        div4 = div3 + div
        self.df.iloc[div3:div4].heave.plot(ax = ax4)

        return fig, (ax1,ax2, ax3,ax4)

    def plot_displacements(self):
        """Plot the 3 displacements

        Returns
        -------
        fig : matplotlib figure obj
            Figure object
        ax : matplotlib axis obj
            Axis object

        """

        fig, (ax1,ax2, ax3) = plt.subplots(3, 1, sharex = True, figsize = (20,10))
    
        self.df.heave.plot(ax = ax1, color = 'green' , title = 'Heave')
        self.df.north.plot(ax = ax2, color = 'blue', title = 'North')
        self.df.west.plot(ax = ax3, color = 'orange', title =  'West')

        return fig, (ax1,ax2, ax3)
        
    def summary(self):
        """prints a summary of the displacement data 
        """
        print("records in dataset: "+str(len(self.heave)))
    
    def sampling_frequency(self):
        """Retrieves the sampling frequency of the displacements.
        
        Parameters
        ----------
        sampling_frequency : ndarray
            time 2 - time1
        
        Returns
        -------
        float : sampling frequency in hz
        
        """
        #sampling frequecy in seconds
        sample_seconds = self.time[1] - self.time[0]
        
        sf = round(1/sample_seconds, 2)
        
        return sf
    
    def crossings(self):
        """Analyses the displacements to calculate the number of waves, periods and times
        
        Parameters
        ----------
        time : nparray
            time array in seconds
        heave : nparray
            heave array
        Returns
        -------
        waveHeights : list
            list wave heights
        wavePeriods : list
            list of wave periods
        times : list
            list of times
        
        """
        
        def pairwise(iterable):
            """
            Yields pairs of items, i.e.
            s -> (s0,s1), (s1,s2), (s2, s3), ...
            Adopted from http://stackoverflow.com/a/5764807
            """
            from itertools import tee
            a, b = tee(iterable)
            next(b, None)
            return zip(a, b)
        
        waveHeights = []
        wavePeriods = []
        times = []
        
        is_positive = self.heave > 0.001  #10mm grace allowed for the zero upcross
        crossings = ( ~is_positive[:-1] & is_positive[1:]).nonzero()[0]
        for waveStart, waveEnd in pairwise(crossings):
            waveDisplacements = self.array[waveStart:waveEnd,:]
            waveHeights.append(max(waveDisplacements[:,2]) - min(waveDisplacements[:,2]))
            wavePeriods.append((self.array[waveEnd, 0]) - (self.array[waveStart, 0]))
            times.append(self.array[waveStart,0])
            
        self.waveHeights = waveHeights
        self.wavePeriods = wavePeriods
            
        return waveHeights, wavePeriods
    
    def plot_waveheights(self):
        """plots ranked wave heights
        
        Returns
        -------
        fig : matplotlib figure obj
            Figure object
        ax : matplotlib axis obj
            Axis object
        """
        waveHeights = sorted(self.waveHeights, reverse=True)
        plt.clf()
        fig , ax = plt.subplots()
        ax.plot(waveHeights)
        plt.title('ranked wave heights')
        plt.ylabel('wave height (m)')
        plt.show()
        
        return fig, ax
        
    
    def hsig(self):
        """Calculates the zero upcrossing signficant wave height.
        
        The signficant wave height, the average wave height from the largest
        1/3 of waves in a given time series.
        
        Parameters
        ----------
        waveHeights : list
            a list containing the wave heights
        
        Returns
        -------
        Hsig : float
        
        """
        
        waveHeights = sorted(self.waveHeights, reverse=True)
        numberofWaves = len(waveHeights)
        HsWaveHeights = waveHeights[:numberofWaves//3]
        hsig = sum(HsWaveHeights) / float(len(HsWaveHeights))
    
        return round(hsig, 3)
    
    def tz(self):
        """The mean wave period, Tz, defined as the aver periods of waves from
        upcrossing analysis.
        
        Returns
        -------
        Tz : float
        
        """
        Tz = sum(self.wavePeriods) / len(self.wavePeriods)
        return round(Tz, 3)
    
    def hmax(self):
        """
        The maximum wave height, Hmax, represents the maximum value of the wave
        height measured over a given period of time.
        
        Returns
        -------
        hmax : float
        """
        return round(max(self.waveHeights), 3)
    
    def h10(self):
        """
        H10 is defined as the average of the largest 1/10 waves in a record. 
        The determination of H10 is based on the zero up-crossing method
           
        Returns
        -------
        h10 : float
        """
        waveHeights = sorted(self.waveHeights, reverse=True)
        numberofWaves = len(waveHeights)
        H10WaveHeights = self.waveHeights[:numberofWaves//10]
        H10 = sum(H10WaveHeights) / float(len(H10WaveHeights))
        return round(H10, 3)
    
    def ts(self):
        """
        The significant wave period, represented by T1/3 or Ts is defined as the
        average period of the largest 1/3 waves in a record
           
        Returns
        -------
        tz : float
        """
        sortedWavePeriods = [x[1] for x in sorted(zip(self.waveHeights,self.wavePeriods), 
                                              key=lambda x: x[0], reverse=True)]
        
        numberofWaves = len(sortedWavePeriods)
        TsWavePeriods = sortedWavePeriods[:numberofWaves//3]
        Ts = sum(TsWavePeriods) / float(len(TsWavePeriods))
        return round(Ts, 3)
        
    def hmean(self):
        """
        Hmean is defined as the average of the wave heights
        
        Returns
        -------
        hmean : float
        """
        Hmean = sum(self.waveHeights) / float(len(self.waveHeights))
        return round(Hmean, 3)

    def tmax(self):
        """
        The maximum wave period, Tmax, represents the maximum value of the
        discrete wave periods measured
        
        Returns
        -------
        tmax : float
        """
        Tmax = float(max(self.wavePeriods))
        return round(Tmax, 3)     
    
    def hrms(self):
        """
        The RMS of wave heights
        
        Returns
        -------
        hrms : float
        """
        numberofWaves = float(len(self.waveHeights))
        squaredWaveHeights = [x**2/numberofWaves for x in self.waveHeights]
        Hrms = sum(squaredWaveHeights) ** 0.5
        return round(Hrms, 3)
        
    def spectra(self, window = None, overlap = True):
        """Calculates a non-directional spectra using welch's method the default 
        uses and overlapping method with a Hann window of 17 segments.
        
        window length based on 4608 records in 30 mins -250 to make 28.4
        
        
        Parameters
        ----------
        heave : nparray
            nparray of the heave data
        sf : float
            sampling frequency of the data
        window : nparray
            window method to use in welch's method when adding a new window 
            the signal processing library for scipy should be used. e.g. 
            scipy.signal.hann
        overlap : Bool
            default value is True however if wanted this can be set to false 
            depending on the window type.
        
        
        Returns
        -------
        f : nparray
            a list of the different frequencies output from the welch's method
            fft calcuation
        Pxx_den : nparray
            a list of power spectral density values for each frequency.
            
        See Also
        --------
        * 'Scipy.Signal <https://docs.scipy.org/doc/scipy/reference/signal.html>'_
        """
        #setup welch method setup we use 17 segments (-250 records to match mk4)
        if self.sampling_frequency() == 2.56:
            windowlen = int(((len(self.heave)-250)/17)*2)
            if overlap == True:
                overlap = int(windowlen // 2)
    
            if window == None:
                window = signal.hann(windowlen)
            heave = self.heave[:-250]
                
        #setup bartletts method using 8 segments non overlapping
        if self.sampling_frequency() == 1.28:
            windowlen = 256
            #cut down displacements to 26.6 minutes
            heave = self.heave[:-262]
            
            #window function
            def datawell_tukey(n, alpha):
                '''
                Custom implementation of the Tukey window as done
                in PB's spreadsheet. Does not use (N-1) terms, just N
                '''
        
                window = np.ones(n)
                
                lower_check = np.divide(alpha*n, 2)
                middle_check = n*(1 - np.divide(alpha, 2)) - 1
                
                for i in range(n):
                    if i < lower_check:
                        window[i] = 0.5 * (1 + np.cos(np.pi * (np.divide(2*i, alpha*n) - 1)))
                    elif i > middle_check:
                        window[i] = 0.5 * (1 + np.cos(np.pi * (np.divide(2*(i + 1), alpha*n) - np.divide(2,alpha) + 1)))
                
                return window
            
            datawell_window = datawell_tukey(windowlen, 0.25)
            window = datawell_window
            
            overlap = None
        f, Pxx_den = signal.welch(self.heave, fs = self.sampling_frequency() ,
                                  window = window , noverlap = overlap)
        
        return f, Pxx_den
        
    def plot_spectra(self, semi = True):
        """Creates a spectrograph of the wave spectra.
        
        Parameters
        ----------
        semi : Bool
            plot style, default is semilogy
        
        Returns
        -------
        fig : matplotlib figure obj
            Figure object
        ax : matplotlib axis obj
            Axis object
        """
        f, Pxx_den = self.spectra()
        plt.clf()
        fig , ax = plt.subplots()
        
        ax.plot(f, Pxx_den)
        plt.title('spectrum')
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        
        return fig, ax

    def plot_3d(self):
        """Creates a 3d plot of the displacements

        Returns
        -------
        fig : matplotlib figure obj
            Figure object
        ax : matplotlib axis obj
            Axis object

        """

        fig = plt.figure(figsize = (10,10))

        ax = fig.add_subplot(4,2,1,projection='3d')
        
        ax.plot3D(self.north,self.west, self.heave, 'gray', linewidth=0.5)
        ax.scatter3D(self.north,self.west, self.heave, 'gray',c=self.heave, cmap = 'viridis', s= 1.5)
        ax.view_init(0, 90)
        plt.xlabel('north')
        plt.ylabel('west')
        
        ax = fig.add_subplot(4,2,2,projection='3d')
        
        ax.plot3D(self.north,self.west, self.heave, 'gray', linewidth=0.5)
        ax.scatter3D(self.north,self.west, self.heave, 'gray',c=self.heave, cmap = 'viridis', s= 1.5)
        ax.view_init(0, 0)
        plt.xlabel('north')
        plt.ylabel('west')
        
        ax = fig.add_subplot(4,2,3,projection='3d')
        
        ax.plot3D(self.north,self.west, self.heave, 'gray', linewidth=0.5)
        ax.scatter3D(self.north,self.west, self.heave, 'gray',c=self.heave, cmap = 'viridis', s= 1.5)
        plt.xlabel('north')
        plt.ylabel('west')
        
        ax = fig.add_subplot(4,2,4,projection='3d')
        
        ax.plot3D(self.north,self.west, self.heave, 'gray', linewidth=0.5)
        ax.scatter3D(self.north,self.west, self.heave, 'gray',c=self.heave, cmap = 'viridis', s= 1.5)
        ax.view_init(90, 0)
        plt.xlabel('north')
        plt.ylabel('west')
        
        plt.tight_layout()
        plt.legend()

        return fig, ax

    def tp(self):
        """
        Peak wave period as defined as the frequency of the peak wave energy.
        converted to seconds.
        
        Returns
        -------
        tp : float
        """
        maxval = np.argmax(self.power)
        tp = 1/(self.f[maxval])
        return round(tp, 3)
        
    def hm0(self):
        """
        Signifcant wave height as calculated from the power spectrum zeorth moment.
        
        Returns
        -------
        hm0 : float
            meters
        """       
        #freq bin width
        freq_width = self.f[1] - self.f[0]
        #energy
        sum0 = np.sum(self.power)
        
        #calculate moment 0
        m0 = sum0 * freq_width
        
        hm0 = 4*np.sqrt(m0)
        
        return round(hm0 , 3)
    
    def t02(self):
        """
        Mean wave period as defined from the spectra.
        
        * uses a trapeziodal method to estimate the moments.
        
        Return
        ------
        t02 : float
            seconds
        """
        
        #calculate moment 0
        m0 = np.trapz(self.power * (self.f ** 0), self.f)
        #calculate moment 2
        m02 = np.trapz(self.power * (self.f ** 2), self.f)
        
        t02 = np.sqrt(m0 /m02)
        
        return round(t02, 3)
    
    def directional_spectra(self):
        """Calculates directional spectra
        
        Step 1: Takeaway 250 from the total (4602), and divide into 17 segments (256 each)
        
        Step 2: For each segement apply tukey window (N terms only) 
        
        Step 3: FFT for each of the three displacements
        
        Step 4: Calculate cross and quadspectra
        
        Step 5: Calculate fourier and centred fourier coefficients
        
        Step 6: Calculate directional spectra parameters
        
        Step 7: Spread directional spectra parameters onto array
        
        Step 8: Average the results for each segment together
        
        
        Returns
        -------
        dir_spec_ave : DataFrame
            Returns a DataFrame object containing directional spectra calulation
            results. This is the average of 17, 256 segements using Bartlett method.
        
        """        
        #convert to dataframe
        df_raw = pd.DataFrame(self.data())
        df_raw = df_raw.drop([0], axis = 1)
        df_raw.columns = ['heave','north', 'west']
        

        # decimate to 1.28hz
        if self.sampling_frequency() == 2.56:
            print('WARNING: downsampling data to 1.28hz')
            df_raw = df_raw.iloc[::2, :]
        
        #split the dataframe in 256 size chunks
        n = 256  #chunk row size
        list_df = [df_raw[i:i+n] for i in range(0,df_raw.shape[0],n)]
        
        #list of direction spectra params
        df_params_list = []
        
        #list of directional spectra arrays
        arr_list = []
        
        for df in list_df:
            
            #window function
            def datawell_tukey(n, alpha):
                '''
                Custom implementation of the Tukey window as done
                in PB's spreadsheet. Does not use (N-1) terms, just N
                '''
        
                window = np.ones(n)
                
                lower_check = np.divide(alpha*n, 2)
                middle_check = n*(1 - np.divide(alpha, 2)) - 1
                
                for i in range(n):
                    if i < lower_check:
                        window[i] = 0.5 * (1 + np.cos(np.pi * (np.divide(2*i, alpha*n) - 1)))
                    elif i > middle_check:
                        window[i] = 0.5 * (1 + np.cos(np.pi * (np.divide(2*(i + 1), alpha*n) - np.divide(2,alpha) + 1)))
                
                return window
            
            datawell_window = datawell_tukey(len(df), 0.25)
            df_window_applied = df.apply(lambda df: df * datawell_window)
            
            #calculate fft
            df_fft = pd.DataFrame()
            
            
            df_fft['heave'] = np.fft.fft(df_window_applied['heave'], axis=-1)
            df_fft['north'] = np.fft.fft(df_window_applied['north'], axis=-1)
            df_fft['west'] = np.fft.fft(df_window_applied['west'], axis=-1)
            
            def cross_periodogram(n, wind_norm, sig_1, sig_2):
                return np.divide(sig_1 * np.conj(sig_2), wind_norm**2 * n**2)
            
            window_norm_sqrt = np.sqrt((datawell_window**2).mean())
            
            #get data length for calculations
            N = len(df)
            
            df_periodogram = pd.DataFrame()
            df_periodogram['hh'] = cross_periodogram(N, window_norm_sqrt, df_fft['heave'], df_fft['heave'])
            df_periodogram['nn'] = cross_periodogram(N, window_norm_sqrt, df_fft['north'], df_fft['north'])
            df_periodogram['ww'] = cross_periodogram(N, window_norm_sqrt, df_fft['west'], df_fft['west'])
            df_periodogram['nw'] = cross_periodogram(N, window_norm_sqrt, df_fft['north'], df_fft['west'])
            df_periodogram['hn'] = cross_periodogram(N, window_norm_sqrt, df_fft['heave'], df_fft['north'])
            df_periodogram['hw'] = cross_periodogram(N, window_norm_sqrt, df_fft['heave'], df_fft['west'])
            
            df_cospectra = df_periodogram[['hh', 'nn', 'ww', 'nw']].apply(lambda x: x.real)
            df_quadspectra = df_periodogram[['hn', 'hw']].apply(lambda x: x.imag)
            
            def fc_a1(Q_hn, C_hh, C_nn, C_ww):
                '''
                a1 Fourier Coefficient
                '''
                return np.divide(Q_hn, np.sqrt(C_hh * (C_nn + C_ww)))
            
            def fc_b1(Q_hw, C_hh, C_nn, C_ww):
                '''
                b1 Fourier Coefficient
                '''
                return np.divide(Q_hw, np.sqrt(C_hh * (C_nn + C_ww)))
            
            def fc_a2(C_nn, C_ww):
                '''
                a2 Fourier Coefficient
                '''
                return np.divide(C_nn - C_ww, C_nn + C_ww)
            
            def fc_b2(C_nn, C_ww, C_nw):
                '''
                b2 Fourier Coefficient
                '''
                return np.divide(C_nw * 2, C_nn + C_ww)
            
            def get_fourier_coefficients(df_cospectra, df_quadspectra):
                '''
                Calculate df with Fourier Coefficients
                '''
                df = pd.DataFrame()
                df['a1']  = fc_a1(df_quadspectra['hn'],
                                  df_cospectra['hh'],
                                  df_cospectra['nn'],
                                  df_cospectra['ww'])
                
                df['b1']  = fc_b1(df_quadspectra['hw'],
                                  df_cospectra['hh'],
                                  df_cospectra['nn'],
                                  df_cospectra['ww'])
                
                df['a2'] = fc_a2(df_cospectra['nn'],
                                 df_cospectra['ww'])
                
                df['b2'] = fc_b2(df_cospectra['nn'],
                                 df_cospectra['ww'],
                                 df_cospectra['nw'])
                
                return df
            
            df_fourier_coefficients = get_fourier_coefficients(df_cospectra[1:int(N/2)], df_quadspectra[1:int(N/2)])
            
            def get_centred_fourier_coefficients(df_fc, df_quadspectra):
                '''
                Calculate centred fourier coefficients from fourier coefficients
                and quadspectra
                '''
                df = pd.DataFrame()
                
                # Direction
                df['dir'] = np.arctan2(df_quadspectra['hw'], df_quadspectra['hn'])
                
                # Centre fourier coefficients
                df['m1'] = np.sqrt(df_fc['a1']**2 + df_fc['b1']**2)
                df['m2'] = df_fc['a2']*np.cos(2 * df['dir']) + df_fc['b2']*np.sin(2 * df['dir'])
                df['n2'] = -1*(-1*df_fc['a2']*np.sin(2 * df['dir']) + df_fc['b2']*np.cos(2 * df['dir']))
                
                # Spread
                df['spread'] = np.sqrt(np.divide(1 - df['m2'], 2))
                
                return df   
            
            df_centred_fc = get_centred_fourier_coefficients(df_fourier_coefficients,
                                                     df_quadspectra[1:int(N/2)])
            
            freq_step = 0.005
            
            def get_dir_spec_params(n, freq_step, cs, c_fc):
                '''
                Calculate directional spectra parameters from:
                n - bins
                freq_step - frequency step
                cs - cospectra parameters
                c_fc - centred fourier coefficients
                '''
                
                df = pd.DataFrame()
            
                
                df['freq'] = np.arange(n/2) * freq_step
                df = df[1:]  # Ignore first zero frequency
                
                df['S'] = np.divide(2*cs['hh'], freq_step)
                df['dir'] = np.mod(np.degrees(np.pi - c_fc['dir']), 360)
                df['spread'] = np.rad2deg(np.sqrt(2 - 2*c_fc['m1']))
                df['skewness'] = np.divide(-1*c_fc['n2'], c_fc['spread']**3)
                df['kurtosis'] = np.divide((6 - 8*c_fc['m1'] + 2*c_fc['m2']),(2 - 2*c_fc['m1'])**2)
                
                #remove data outside frequency bands of interest
                df = df[(df['freq'] >= 0.025) &  (df['freq'] <= 0.58)]
                
                return df
            
            df_dir_spec = get_dir_spec_params(N, freq_step, df_cospectra[1:int(N/2)], df_centred_fc)
            
            df_params_list.append(df_dir_spec)
    
            def truncated_fourier(S_f, direction, a1, a2, b1, b2):
                return S_f * (1/(2*np.pi) + (1/np.pi) * ( \
                                                         a1*np.cos(np.pi - direction) +\
                                                         b1*np.sin(np.pi - direction) +\
                                                         a2*np.cos(2*(np.pi - direction)) +\
                                                         b2*np.sin(2*(np.pi - direction))
                                                         ))
            
            def get_dir_spec_arr(df_dir_spec, df_fc, del_deg=10):
                '''
                Calculate array of S(f,theta) (directional wave spec)
                
                Spreads the wave spectra for a particular frequency across
                wave direction
                '''
                direction = np.deg2rad(np.arange(0, 370, del_deg))
                arr = np.zeros((len(df_dir_spec), len(direction)))
                for ind, (ds, fc) in enumerate(zip(df_dir_spec.iterrows(), df_fourier_coefficients.iterrows())):
                    for j, d in enumerate(direction):
                        arr[ind,j] = truncated_fourier(ds[1]['S'], d, fc[1]['a1'], fc[1]['a2'], fc[1]['b1'], fc[1]['b2'])
        
                return arr
                
            arr = get_dir_spec_arr(df_dir_spec, df_fourier_coefficients)
            arr_list.append(arr)
            
        #average all 17 segments together
        dir_spec_ave = pd.concat(df_params_list).groupby(['freq']).mean()
        
        #average all 17 arrays together
        dir_spec_arr = np.average(arr_list, axis = 0)

        return dir_spec_ave, dir_spec_arr
    
    def plot_directional_spectra(self):
        """
        Creates a polar plot of the directional spectra
        
        Returns
        -------
        ax : matplotlib axis obj
            Axis object
        
        """
        
        #negative values can be produced in this directional spectrum method these will be truncated to 0
        spectra_array = self.dir_spectra_array
        spectra_array = spectra_array.clip(min=0.1)
        #mask array make zero values NaN
        spectra_array = np.ma.masked_equal(spectra_array, 0)
        
        F, D = np.meshgrid(np.deg2rad(np.arange(0, 370, 10)), 1/self.dir_spectra.index.values)
        ax = plt.subplot(111, projection='polar')
        #ax = fig.gca(projection='polar')
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        cmap = cm.rainbow
        cmap.set_under(color = 'white')
        CS = ax.contourf(F,D, spectra_array, cmap =cmap, vmin = 0.00001, extend = 'both')
        ax.set_ylim(0, 25)
        
        cbar = plt.colorbar(CS)
        cbar.ax.set_ylabel('m$^2$s/deg')
        
        return ax
        
    def pkDir(self):
        """
        Peak direction calculated as the wave direction with the maximum energy
        
        """
        dirspec = self.dir_spectra
        
        
        return round(dirspec[dirspec.S == dirspec.S.max()].dir.values[0], 3)
        
    
    def meanDir(self):
        """
        Mean wave direction 
        """
        dirspec = self.dir_spectra
        
        return round(dirspec.dir.mean(), 3)
    
    def wave_stats(self):
        """
        Wave Statistics summary
        
        Returns
        -------
        wave stats : DataFrame
            a dataframe of the summary wave statistics
        
        """
        nameList = ['hsig','hmax','tz','tp','hm0','hrms','t02', 'pkDir', 'aveDir']
        sumList = [self.hsig(), self.hmax(), self.tz(), self.tp(), self.hm0(),
                   self.hrms(), self.t02(), self.pkDir(), self.meanDir()]
        summary = pd.DataFrame(sumList)
        summary.index = nameList
        
        return summary
    

    
    
class filtered_displacements(displacements):
    """
    Filtered displacement data class.
        
    A sub of the displacement class which provides a filter methods
    
    Inheritance
    -----------
    displacements : parent
        inherits from the displacements parent class for methods and objects
        
    Methods
    -------
    butterworth : method
        when called modifies the baseline data object filetering the data
        
    """
    
    def butterworth(self, lowcutoff = 40):
        """
        Perform butterworth filter on each of the different displacements
        
        WARNING: this function modifies the original object
        
        Parameters
        ----------
        lowcutoff : int
            the cuttoff frequency you wish to use in seconds
            default value is 40 seconds
        
        """
        
        
        def butter_bandpass(lowcut, fs, order=5):
            """Butterworth bandpass filter construction
    
            Args:
                arg1 (int): The cutoff frequency in Hz 
                arg2 (float): The sampling frequency of the data
                arg3 (int): The order used on the window

            Returns:
                b (nparray): window object
                a (nparray): window object

            """
            nyq = 0.5 * fs
            low = lowcut / nyq
            #high = highcut / nyq
            b, a = butter(order, low, btype='highpass', analog = False, output='ba')
            return b, a      
            
        def butter_bandpass_filter(data, lowcut, fs, order=5):
            """
            Application of the bandpass filter with our data currently filtered to 40 seconds
        
            Args
            ----
            arg1 (nparray): Numpy array of the displacement data 
                arg2 (int): Frequency of cut off
                arg3 (float): Float of the sampling frequency of the data
                arg3 (int): The order of the butterworth cutoff window
        
            Returns:
                nparray: filtered heave data with the butterworth filter applied.
        
                    b, a = butter_bandpass(lowcut, fs, order=order)
                    y = lfilter(b, a, data)
                    return y
            """
            b, a = butter_bandpass(lowcut, fs, order=order)
            y = filtfilt(b, a, data, method = 'gust')
                
            return y
        
        def butterFilter(array, f, lowcutoff):
            """
            A wrapper function that gathers the relevant information 
            for butterworth filter construction
        
            Arguements
            ----------
            array : ndarray
                Numpy array of displacement data
            f : float
                the sampling frequency
            lowcutoff : int
                the low frequency cutoff point in seconds
                
        
            Returns
            -------
            filt: ndarray
                Numpy array filtered using a butterworth filter
            """
        
            


            element = array[:, 1]

            fs = f  #the frquency of the wavebuoy
            lowcut = 1 / lowcutoff
       
        
            array[:,1] = butter_bandpass_filter(element, lowcut, fs, order=5)
        
            return array
        
        #modify original data objects to ensure revised are correct
        self.array = butterFilter(self.data(), self.sampling_frequency(), lowcutoff)
        self.crossings()
        self.f , self.power = self.spectra()
    
        return 
    
    def plot_butterworth(self, lowcutoff=40):
        """
        Plot a butterworth filter
        
        Parameters
        ----------
        lowcutoff : int
            the low frequency cutoff point in seconds
        
        """
        def _butter_bandpass(lowcut, fs, order=5):

            nyq = 0.5 * fs
            low = lowcut / nyq
            #high = highcut / nyq
            b, a = butter(order, low, btype='highpass', analog = False, output='ba')
            return b, a      
        
        
        def butterFilter(array, f, lowcutoff):
            fs = f  #the frquency of the wavebuoy
            lowcut = 1 / lowcutoff
            
            b, a = _butter_bandpass(lowcut, fs, order=5)
            w, h = signal.freqz(b, a)
        
            plt.clf()
            fig , ax = plt.subplots()
            ax.plot((fs * 0.5 / np.pi) * w, abs(h))
            ax.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
               '--', label='sqrt(0.5)')
            plt.ylabel('Gain')
            plt.grid(True)
            plt.legend(loc='best')          
            
            return  fig, ax
        
        fig, ax = butterFilter(self.data(), self.sampling_frequency(), lowcutoff)
        
        return fig, ax
    
    def spike_removal(self, standard_deviations = 5):
        """
        A coarse spike removal filter that removes large spikes in the data
        
        WARNING: This method modifies the original object
        
        Parameters
        ----------
        standard_deviations : int
            the number of standard deviations to use when applying the filter
            default value is 5
        """
        def remove(array, standard_deviations):
                       
            #loop through the different heave dimensions
            for dimension in range(1,4):
                   
                #calculate_std for data
                d_std = np.std(array[:,dimension])
            
                #get mean heave
                d_mean = np.mean(array[:,dimension])
                
                #create our error threshold
                errorthreshold = (d_mean+(d_std*standard_deviations))
                
                #get spikes, we may need these later
                spikes = array[array[:, dimension] > ((d_mean+d_std)*standard_deviations)]
                
                #modify the data, changing data over the threshold to the mean
                array[array[:,dimension] > (errorthreshold)] = d_mean
            return array
        
        self.array = remove(self.data(), standard_deviations)
        self.crossings()
        self.f , self.power = self.spectra()
                            
        return

class filtered_displacements_1d(displacements_1d):
    """
    Filtered displacement data class.
        
    A sub of the displacement class which provides a filter methods
    
    Inheritance
    -----------
    displacements_1d : parent
        inherits from the displacements_1d parent class for methods and objects
        
    Methods
    -------
    butterworth : method
        when called modifies the baseline data object filetering the data
        
    """
    
    def butterworth(self, lowcutoff = 40):
        """
        Perform butterworth filter on each of the different displacements
        
        WARNING: this function modifies the original object
        
        Parameters
        ----------
        lowcutoff : int
            the cuttoff frequency you wish to use in seconds
            default value is 40 seconds
        
        """
        
        
        def butter_bandpass(lowcut, fs, order=5):
            """Butterworth bandpass filter construction
    
            Args:
                arg1 (int): The cutoff frequency in Hz 
                arg2 (float): The sampling frequency of the data
                arg3 (int): The order used on the window

            Returns:
                b (nparray): window object
                a (nparray): window object

            """
            nyq = 0.5 * fs
            low = lowcut / nyq
            #high = highcut / nyq
            b, a = butter(order, low, btype='highpass', analog = False, output='ba')
            return b, a      
            
        def butter_bandpass_filter(data, lowcut, fs, order=5):
            """
            Application of the bandpass filter with our data currently filtered to 40 seconds
        
            Args
            ----
            arg1 (nparray): Numpy array of the displacement data 
                arg2 (int): Frequency of cut off
                arg3 (float): Float of the sampling frequency of the data
                arg3 (int): The order of the butterworth cutoff window
        
            Returns:
                nparray: filtered heave data with the butterworth filter applied.
        
                    b, a = butter_bandpass(lowcut, fs, order=order)
                    y = lfilter(b, a, data)
                    return y
            """
            b, a = butter_bandpass(lowcut, fs, order=order)
            y = filtfilt(b, a, data, method = 'gust')
                
            return y
        
        def butterFilter(array, f, lowcutoff):
            """
            A wrapper function that gathers the relevant information 
            for butterworth filter construction
        
            Arguements
            ----------
            array : ndarray
                Numpy array of displacement data
            f : float
                the sampling frequency
            lowcutoff : int
                the low frequency cutoff point in seconds
                
        
            Returns
            -------
            filt: ndarray
                Numpy array filtered using a butterworth filter
            """
        
            dimension = 1

            element = array[:, dimension]

            fs = f  #the frquency of the wavebuoy
            lowcut = 1 / lowcutoff
       

            array[:,dimension] = butter_bandpass_filter(element, lowcut, fs, order=5)

            return array
        
        #modify original data objects to ensure revised are correct
        self.array = butterFilter(self.data(), self.sampling_frequency(), lowcutoff)
        self.crossings()
        self.f , self.power = self.spectra()
    
        return 
    
    def plot_butterworth(self, lowcutoff=40):
        """
        Plot a butterworth filter
        
        Parameters
        ----------
        lowcutoff : int
            the low frequency cutoff point in seconds
        
        """
        def _butter_bandpass(lowcut, fs, order=5):

            nyq = 0.5 * fs
            low = lowcut / nyq
            #high = highcut / nyq
            b, a = butter(order, low, btype='highpass', analog = False, output='ba')
            return b, a      
        
        
        def butterFilter(array, f, lowcutoff):
            fs = f  #the frquency of the wavebuoy
            lowcut = 1 / lowcutoff
            
            b, a = _butter_bandpass(lowcut, fs, order=5)
            w, h = signal.freqz(b, a)
        
            plt.clf()
            fig , ax = plt.subplots()
            ax.plot((fs * 0.5 / np.pi) * w, abs(h))
            ax.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
               '--', label='sqrt(0.5)')
            plt.ylabel('Gain')
            plt.grid(True)
            plt.legend(loc='best')          
            
            return  fig, ax
        
        fig, ax = butterFilter(self.data(), self.sampling_frequency(), lowcutoff)
        
        return fig, ax
    
    def spike_removal(self, standard_deviations = 5):
        """
        A coarse spike removal filter that removes large spikes in the data
        
        WARNING: This method modifies the original object
        
        Parameters
        ----------
        standard_deviations : int
            the number of standard deviations to use when applying the filter
            default value is 5
        """
        def remove(array, standard_deviations):
                       
            #loop through the different heave dimensions
            for dimension in range(1,4):
                   
                #calculate_std for data
                d_std = np.std(array[:,dimension])
            
                #get mean heave
                d_mean = np.mean(array[:,dimension])
                
                #create our error threshold
                errorthreshold = (d_mean+(d_std*standard_deviations))
                
                #get spikes, we may need these later
                spikes = array[array[:, dimension] > ((d_mean+d_std)*standard_deviations)]
                
                #modify the data, changing data over the threshold to the mean
                array[array[:,dimension] > (errorthreshold)] = d_mean
            return array
        
        self.array = remove(self.data(), standard_deviations)
        self.crossings()
        self.f , self.power = self.spectra()
                            
        return
  