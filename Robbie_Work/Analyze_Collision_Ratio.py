# Import all necessary libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from operator import itemgetter

amu_to_kg = 1.66054*(10**-27)
joules_to_eV = 6.242*(10**18)


def gaussian(x, amplitude, center, stddev):
    """
    Given all necessary data, find the gaussian fit using the gaussian formula: 

    g(x) = ae^(-(x-c)/b)^2
    where a is the amplitude
    c is the center
    b is the width, = sqrt(2)stddev
    and g(x) is a function of x

    """

    gassian_fit = amplitude*np.exp(-(((x-center)**2) /(2* (stddev**2))))
    return gassian_fit

def fit_gaussian (xdata, ydata, peak_index, number_of_points, width_guess):
    """
    Params:
    xdata -> energy 
    ydata -> Frequency
    peak_index -> Which energy is the most frequent 
    number_of_point -> How many energys to be considered when fitting (bin size)
    width_guess -> Estimate for stddev of energys and frequencies considered
    """

    # Ensure no indexing out of bounds 
    start_index = max(0, peak_index - number_of_points)
    end_index = min(len(xdata), peak_index + number_of_points + 1)    

    #data for fitting, in specific bin size 
    energy = xdata[start_index:end_index]
    frequency = ydata[start_index:end_index]

    # Need guess to initially fit data, assumption is that the peak frequency will correlate to the peak of the gaussian fit, 
    # also assuming that the center of gaussian fit will be at the same energy level for the maximum intensity
    amplitude_guess = ydata[peak_index]
    center_guess = xdata[peak_index]

    # Try to fit curve using the gaussian function where input parameters are the energy levels, frequency and guesses for gaussian fitting
    try:
        popt, _ = curve_fit(gaussian, energy, frequency, p0=[amplitude_guess, center_guess, width_guess])
    except Exception as e:
        print(f"Error fitting Gaussian at peak {peak_index}: {e}")
        return energy, np.zeros_like(energy)

    popt[0] = amplitude_guess  

    return energy, gaussian(energy, *popt)

def fit_data (speeds, expected_outs, number_of_points=25,width_guess=0.5):
    """
    params:
        files: integers of the speed [km/s] of given test
        expected_outs: NIST values for expected observed spectra
        number_of_points: How large the bin size is for the gaussian fit, how many points to consider when fitting
        width_gess: A quick estimate on to what the stddev is for the collected data, a guess is used to prevent having to do
            histogramic calculations, and can maintain data in its point-like form
    """
    PEAK_PROMINENCE = 0.01
    search_range = 15.0  
    all_peaks =[]

    ratio_data = []
    for i, speed in enumerate(speeds):
        filename = f'../bareNeonHe/Ne10He{str(speed)}kms.csv'
        #Holds peaked calculated from gaussian fits
        file_peaks = []

        #If file exists create dataframe of all data in file
        try:
            data = pd.read_csv(filename)
        except FileNotFoundError:
            print(f"{filename} not found.")
            continue

        energy = data['x'].values
        frequency = data['y'].values

        for i, center in enumerate(expected_outs):
            lower = center-search_range
            upper = center+search_range
            #Find bounds based on search range parameter
            indices_in_range = np.where((energy >= lower) & (energy <= upper))[0] 

            peaks_in_range, _ = find_peaks(frequency[indices_in_range], prominence=PEAK_PROMINENCE)
            largest_peak_index_in_range = peaks_in_range[np.argmax(frequency[indices_in_range][peaks_in_range])]
            largest_peak_index = indices_in_range[largest_peak_index_in_range]

            x_fit, y_fit = fit_gaussian(energy, frequency, largest_peak_index, number_of_points, width_guess)

            y_peak = np.max(y_fit)
            file_peaks.append(y_peak)
        all_peaks.append(file_peaks)

    return x_fit, y_fit, all_peaks

def plot (speeds,expected_outs, number_of_points=25, width_guess=0.5, HCI="O"): 
    """
    params:
        files: integers of the speed [km/s] of given test
        expected_outs: NIST values for expected observed spectra
        number_of_points: How large the bin size is for the gaussian fit, how many points to consider when fitting
        width_gess: A quick estimate on to what the stddev is for the collected data, a guess is used to prevent having to do
            histogramic calculations, and can maintain data in its point-like form
    """
    #Set up the graphical outputs using pyplot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))
    colors = ['blue', 'green', 'red', 'orange', 'purple'] 

    # What peak are we looking for, how far are we searching for it
    PEAK_PROMINENCE = 0.01
    search_range = 5
    
    #Used for determining lyman series on graph
    ratio_out_titles = ['beta','gamma','delta','epsilon']

    # Holder for future data
    ratio_data = []
    for i, speed in enumerate(speeds):
        filename = f'../bareNeonHe/Ne10He{str(speed)}kms.csv'
        
        #Holds peaked calculated from gaussian fits
        peaks = []

        #If file exists create dataframe of all data in file
        try:
            data = pd.read_csv(filename)
        except FileNotFoundError:
            print(f"{filename} not found.")
            continue
        

        energy = data['x'].values
        frequency = data['y'].values
        color = colors[i % len(colors)]  

        # For Each NIST expected energy output
        for i, center in enumerate(expected_outs):
            lower = center-search_range
            upper = center+search_range

            # Plot graphs of pure data
            if i==1:
                ax1.plot(energy, frequency, label=f'Original Data (Speed {speed} km/s)', marker='o', linestyle='-', color=color)
            else:  
                ax1.plot(energy, frequency, label=f'', marker='o', linestyle='-', color=color)
                

            indices_in_range = np.where((energy >= lower) & (energy <= upper))[0] #Find bounds based on search range parameter
            peaks_in_range, _ = find_peaks(frequency[indices_in_range] ,prominence=PEAK_PROMINENCE)
            #TODO: Fix 
            
            #print(indices_in_range)
            #print(frequency[indices_in_range])
            #print(peaks_in_range)
            #print(frequency[indices_in_range][peaks_in_range])
            try:
                largest_peak_index_in_range = peaks_in_range[np.argmax(frequency[indices_in_range][peaks_in_range])]
                largest_peak_index = indices_in_range[largest_peak_index_in_range]

                x_fit, y_fit = fit_gaussian(energy, frequency, largest_peak_index, number_of_points, width_guess)
            
                ax2.plot(x_fit, y_fit, label=f'Fitted Gaussian (Speed {speed} km/s)'if i==1 else "", linestyle='--', color=color)
                y_peak = np.max(y_fit)
                peaks.append(y_peak)
            except Exception as e:
                print(e)
                print(f"No Peak Found at index {indices_in_range}")
                continue
        
        #print(peaks)
        for i in range(1,len(peaks)):
            #print(i)
            ratio_data.append(peaks[i]/peaks[0])
        
    con_speeds = []
    for i, speed in enumerate(speeds):
        new_speed = (speed**2)*0.5*amu_to_kg*joules_to_eV*(10**3)
        
        con_speeds.append(new_speed)

    # print(con_speeds)
    cycle = len(expected_outs)-1


    for i in range(0,cycle):
        color = colors[i % len(colors)]
        try:
            ax3.plot(speeds,itemgetter(i,cycle+i,2*cycle+i)(ratio_data),label=f'{HCI} Lyman-{ratio_out_titles[i]}', marker='o', linestyle='--', color=color)
        except Exception as e:
            print(e)
            continue

    # Graph Setup for Visuals
    ax1.set_title('Raw Data for Different Speeds and Energy Intervals')
    ax1.set_xlabel('x (Energy eV)')
    ax1.set_ylabel('y (Intensity)')

    ax2.set_title('Data with Fitted Gaussians for Different Speeds and Energy Intervals')
    ax2.set_xlabel('x (Energy eV)')
    ax2.set_ylabel('y (Intensity)')


    ax3.set_title('Collision Ratios vs Input Energy')
    ax3.set_xlabel('Input Energy [keV/amu]')
    ax3.set_ylabel('Ratio Data Intensity 2 / Intensiity 1 ')

    ax1.legend()
    ax1.grid(True)
    ax2.legend()
    ax2.grid(True)
    ax3.legend()
    ax3.grid(True)
    ax1.set_xlim(left=0)
    ax1.set_xlim(right=np.max(x_fit))
    ax2.set_xlim(left=0)
    ax3.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    ax3.set_ylim(bottom=0)
    plt.show()


speeds = [350,440,647]

expected_outs_OVIII = [653.5,774.5,817.0,836.5,847.2]
expected_outs_NVII = [500.0,593.0,625.5,640.5]
expected_outs_NeX = [1021.5, 1211.0, 1277.0, 1308.0]
expected_outs_CVI = [367.5,435.5,459.5,470.5,476.5]
plot(speeds,expected_outs_OVIII,HCI="OVIII")