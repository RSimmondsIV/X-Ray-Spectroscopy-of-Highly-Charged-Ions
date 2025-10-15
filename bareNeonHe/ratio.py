import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

def gaussian(x, amp, cen, wid):
    return amp * np.exp(-((x - cen) / wid)**2)

def fit_gaussian_to_peak(x, y, peak_index, num_points=30, wid_guess=4.0):  
    start_index = max(0, peak_index - num_points)
    end_index = min(len(x), peak_index + num_points + 1)

    x_fit = x[start_index:end_index]
    y_fit = y[start_index:end_index]

    amp_guess = y[peak_index]  
    cen_guess = x[peak_index]  
    wid_guess = wid_guess      

    bounds = ([0, x_fit[0] - 0.1, 0.1], [np.inf, x_fit[-1] + 0.1, np.inf])
    maxfev = 10000  

    try:
        popt, _ = curve_fit(gaussian, x_fit, y_fit, p0=[amp_guess, cen_guess, wid_guess], bounds=bounds, maxfev=maxfev)
    except Exception as e:
        print(f"Error fitting Gaussian at peak {peak_index}: {e}")
        return x_fit, np.zeros_like(x_fit)

    popt[0] = amp_guess  

    return x_fit, gaussian(x_fit, *popt)

def plot_data_and_fit(speeds, num_points=5, wid_guess=2.0, peak_prominence=0.01):
    energy_centers = [1021.5, 1211.0, 1277.0, 1308.0]
    search_range = 15.0  

    plt.figure(figsize=(10, 6))
    colors = ['blue', 'green', 'red', 'orange', 'purple']  

    # Table to store the results, with columns for speed and peaks
    results_table = pd.DataFrame(columns=["Speed", "Peak 1", "Peak 2", "Peak 3", "Peak 4"])
    ratio_data = []  # List to store speed vs ratio (Peak 2 / Peak 1)

    for idx, speed in enumerate(speeds):
        file_path = f'Ne10He{speed}kms.csv'
        try:
            data = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"File for speed {speed} not found.")
            continue

        x = data['x'].values  
        y = data['y'].values  

        row_data = [speed]  # Start row with the speed
        color = colors[idx % len(colors)]  

        plt.plot(x, y, label=f'Original Data (Speed {speed} km/s)', marker='o', linestyle='-', color=color)

        peak_1_y = None
        peak_2_y = None

        for i, center in enumerate(energy_centers):
            lower_bound = center - search_range
            upper_bound = center + search_range
            indices_in_range = np.where((x >= lower_bound) & (x <= upper_bound))[0]

            if len(indices_in_range) == 0:
                print(f"No data found in the interval [{lower_bound}, {upper_bound}] for speed {speed}")
                row_data.append(None)  # No peak found, append None
                continue

            peaks_in_range, _ = find_peaks(y[indices_in_range], prominence=peak_prominence)

            if len(peaks_in_range) == 0:
                print(f"No peaks found in the interval [{lower_bound}, {upper_bound}] for speed {speed}")
                row_data.append(None)  # No peak found, append None
                continue

            largest_peak_index_in_range = peaks_in_range[np.argmax(y[indices_in_range][peaks_in_range])]
            largest_peak_index = indices_in_range[largest_peak_index_in_range]

            x_fit, y_fit = fit_gaussian_to_peak(x, y, largest_peak_index, num_points, wid_guess)
            plt.plot(x_fit, y_fit, label=f'Fitted Gaussian {i+1} (near {center} eV, Speed {speed} km/s)', linestyle='--', color=color)

            max_gaussian_y = np.max(y_fit)  # The y-value at the peak of the Gaussian

            print(f"Peak {i+1} maximum near {center} eV for speed {speed}: y = {max_gaussian_y:.4f}")

            # Store the peak y-value in the row data
            row_data.append(max_gaussian_y)

            # Keep track of Peak 1 and Peak 2 y-values for the ratio calculation
            if i == 0:
                peak_1_y = max_gaussian_y
            elif i == 1:
                peak_2_y = max_gaussian_y

        # Add row data to the results table
        results_table.loc[len(results_table)] = row_data

        # Calculate the ratio (Peak 2 / Peak 1) if both are found
        if peak_1_y is not None and peak_2_y is not None:
            ratio = peak_2_y / peak_1_y
            ratio_data.append((speed, ratio))

    # Finalize the plot
    plt.title('Data with Fitted Gaussians for Different Speeds and Energy Intervals')
    plt.xlabel('x (Energy eV)')
    plt.ylabel('y (Intensity)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot the speed vs ratio graph
    if ratio_data:
        ratio_speeds, ratios = zip(*ratio_data)
        plt.figure(figsize=(8, 5))
        plt.plot(ratio_speeds, ratios, marker='o', linestyle='', color='magenta')  # Set linestyle='' to remove the connecting line
        plt.title('Speed vs Ratio (Peak 2 / Peak 1)')
        plt.xlabel('Speed (km/s)')
        plt.ylabel('Ratio (Peak 2 / Peak 1)')
        plt.grid(True)
        plt.show()


    # Output the table of results
    print("\nResults table:")
    print(results_table)

    # Optionally, save the table to a CSV file
    results_table.to_csv("gaussian_fit_results.csv", index=False)

def changespeed(string, old_speed, new_speed):
    return string.replace(str(old_speed), str(new_speed))

# Input multiple speeds of your choice
speeds = input("Enter the speeds you'd like to analyze, separated by commas (e.g., 350,440,647): ").split(',')
speeds = [int(speed.strip()) for speed in speeds]  # Convert input to a list of integers

# Call the plot function for all speeds
plot_data_and_fit(speeds, peak_prominence=0.005)
