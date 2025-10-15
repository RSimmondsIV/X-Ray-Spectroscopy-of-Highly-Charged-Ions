import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import os

def gaussian(x, amp, cen, wid):
    return amp * np.exp(-((x - cen) / wid)**2)

def fit_gaussian_to_peak(x, y, peak_index, num_points=30, wid_guess=4.0):  
    start_index = max(0, peak_index - num_points)
    end_index = min(len(x), peak_index + num_points + 1)

    x_fit_original = x[start_index:end_index]
    y_fit = y[start_index:end_index]

    amp_guess = y[peak_index]  
    cen_guess = x[peak_index]  
    wid_guess = wid_guess      

    bounds = ([0, x_fit_original[0] - 0.1, 0.1], [np.inf, x_fit_original[-1] + 0.1, np.inf])
    maxfev = 10000  

    try:
        popt, _ = curve_fit(gaussian, x_fit_original, y_fit, p0=[amp_guess, cen_guess, wid_guess], bounds=bounds, maxfev=maxfev)
    except Exception as e:
        print(f"Error fitting Gaussian at peak {peak_index}: {e}")
        return x_fit_original, np.zeros_like(x_fit_original), None

    amp, cen, wid = popt
    x_fit_smooth = np.linspace(x_fit_original[0], x_fit_original[-1], 200)
    y_fit_smooth = gaussian(x_fit_smooth, amp, cen, wid)

    return x_fit_smooth, y_fit_smooth, popt

def plot_data_and_fit(first_input, speeds, num_points=5, wid_guess=2.0, peak_prominence=0.01):
    energy_centers = [1021.5, 1211.0, 1277.0, 1308.0]
    search_range = 15.0  

    colors = ['blue', 'green', 'red', 'orange']  

    columns = ["Speed"]
    for i in range(1, 5):
        columns.extend([f"Peak {i}", "Exp. Center", "Center", "Width"])
    results_table = pd.DataFrame(columns=columns)
    
    ratio_data = []  
    plt.figure(figsize=(10, 10))

    for i, speed in enumerate(speeds):
        filename = f"{first_input}{speed}kms.csv"
        if not os.path.isfile(filename):
            print(f"File {filename} not found.")
            continue

        data = pd.read_csv(filename, skiprows=1)
        x = data.iloc[:, 0].dropna().values  
        y = data.iloc[:, 1].dropna().values  
        
        row_data = [speed]  
        color = colors[i % len(colors)]  

        min_len = min(len(x), len(y))
        x, y = x[:min_len], y[:min_len]

        # Calculate histogram bar width based on x value differences
        if len(x) > 1:
            bar_width = abs(x[1] - x[0])  # Absolute value of the difference between consecutive x values
        else:
            bar_width = 1.0  

        plt.subplot(2, 1, 1)
        plt.bar(x, y, color=color, alpha=0.6, label=f'Histogram (Speed {speed} km/s)', width=bar_width)
        plt.ylabel('Intensity')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        peak_1_y, peak_2_y = None, None

        for j, center in enumerate(energy_centers):
            lower_bound = center - search_range
            upper_bound = center + search_range
            indices_in_range = np.where((x >= lower_bound) & (x <= upper_bound))[0]

            if len(indices_in_range) == 0:
                row_data.extend([None, center, None, None])
                continue

            peaks_in_range, _ = find_peaks(y[indices_in_range], prominence=peak_prominence)
            if len(peaks_in_range) == 0:
                row_data.extend([None, center, None, None])
                continue

            largest_peak_index_in_range = peaks_in_range[np.argmax(y[indices_in_range][peaks_in_range])]
            largest_peak_index = indices_in_range[largest_peak_index_in_range]

            x_fit_smooth, y_fit_smooth, popt = fit_gaussian_to_peak(x, y, largest_peak_index, num_points, wid_guess)
            plt.plot(x_fit_smooth, y_fit_smooth, color=color, linewidth=1.5)

            if popt is not None:
                amp, cen, wid = popt
                row_data.extend([amp, center, cen, wid])
                if j == 0:
                    peak_1_y = amp
                elif j == 1:
                    peak_2_y = amp
            else:
                row_data.extend([None, center, None, None])

        results_table.loc[len(results_table)] = row_data

        if peak_1_y is not None and peak_2_y is not None:
            ratio = peak_2_y / peak_1_y
            ratio_data.append((speed, ratio))

        # Add each speed to legend only once on the bottom plot
        plt.subplot(2, 1, 2)
        plt.plot([], [], label=f'Speed {speed} km/s', color=color) 

    plt.subplot(2, 1, 2)
    plt.title('Fitted Gaussians for Different Speeds and Energy Intervals')
    plt.xlabel('Energy (eV)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    if ratio_data:
        ratio_speeds, ratios = zip(*ratio_data)
        plt.figure(figsize=(8, 5))
        plt.plot(ratio_speeds, ratios, marker='o', linestyle='', color='magenta')  
        plt.title('Speed vs Ratio (Peak 2 / Peak 1)')
        plt.xlabel('Speed (km/s)')
        plt.ylabel('Ratio (Peak 2 / Peak 1)')
        plt.grid(True)
        plt.show()

    print("\nResults table:")
    print(results_table)
    results_table.to_csv("gaussian_fit_results.csv", index=False)

# Main execution
first_input = input("Enter an ion and neutral gas (e.g., Ne10He): ")
speeds_input = input("Enter speeds (comma-separated, e.g., 440, 550): ")
speeds = [int(s.strip()) for s in speeds_input.split(",")]

plot_data_and_fit(first_input, speeds, peak_prominence=0.005)
