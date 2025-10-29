import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


def gausian(x,amplitude,center,width):
    """
    Given all necessary data, find the gaussian fit using the gaussian formula: 

    g(x) = ae^(-(x-c)/b)^2

    where a is the amplitude
    c is the center
    b is the width, = sqrt(2)stddev
    and g(x) is a function of x

    """
    gassian_fit = amplitude*np.exp(-((x-center) / width) **2 )
    return gassian_fit

def fit_gaussian(x,y, peak, point_amount, width_guess):
    """
    Using scipy's curve_fit function, 
    fit the gaussian of each peak with bounds and estimated values both of which are taken to be known
    """

    # only data above 0 is important here, thus any data below 0, meaning less than 0 energy is ignored.
    start_index = max(0,peak-point_amount)

    # only data within what was collected can be used here, any data beyond energy data that was gathered is ignored
    end_index = min(len(x),peak+point_amount+1)

    # Energies that were measured for
    Energy_range = x[start_index:end_index]

    # The amount of times each energy was observed
    Frequency= y[start_index:end_index]

    # The Energy and Frequency of the peak value are 
    amplitude_guess = y[peak]
    center_guess = x[peak]
    width_guess = width_guess

    #Inherent Characteristic necessary for scipy.optimize curvefit
    #Set to be greater than 0
    bounds = [0,Energy_range[0]-0.1, 0.1], [np.inf, Energy_range[-1] + 0.1, np.inf]

    # Try to fit the data given the known values of Energy being considered and the amount of times each of those Energies was measured
    # Given a guess of the x,y,std of the peak value (most frequently occuring Energy) fit the data.

    try:
        popt = curve_fit(gausian, Energy_range, Frequency, p0=[amplitude_guess,center_guess,width_guess],bounds=bounds)
    except Exception as e:
        print(f'Error in fitting data at {peak}:{e}')
        return Energy_range, np.zeros_like(Energy_range)

    return popt

def plot_data(speeds, point_amount, width_guess, peak_prominence):
    """
    Given known values of speeds of e- beam, plot the energy peaks of recorded xrays

    params:
        speed of known e- beam, necessary for collision energies

    """

    #Expected values from theory for NeXHe
    energy_centers = [1021.5, 1211.0, 1277.0, 1308.0]
    search_range = 15 #TODO - WHAT IS THIS

    results_df = pd.DataFrame(columns=["Speed","Peak1","Peak2", "Peak3,", "Peak4"])
    #stores (Peak2/Peak1)
    speed_ratio = [] 

    for index, speed in enumerate(speeds):
        #find file, must follow below naming convention
        
        path = f"Ne10He{speed}kms.csv"
        

