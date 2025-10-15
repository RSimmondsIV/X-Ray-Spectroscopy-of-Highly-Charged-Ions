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
        return x_fit_original, np.zeros_like(x_fit_original)

    # popt[0] = amp_guess  
    print(popt)

    # Create a smoother x_fit for a refined curve
    x_fit_smooth = np.linspace(x_fit_original[0], x_fit_original[-1], 200)
    y_fit_smooth = gaussian(x_fit_smooth, *popt)

    return x_fit_smooth, y_fit_smooth

def plot_data_and_fit(filename, num_points=5, wid_guess=2.0, peak_prominence=0.01):
    energy_centers = [1021.5, 1211.0, 1277.0, 1308.0]
    search_range = 15.0  

    data = pd.read_csv(filename, skiprows=1)  # Skip the first row

    speeds = [440, 376, 647, 350]  # Adjusted speeds
    colors = ['blue', 'green', 'red', 'orange']  

    # Table to store the results
    results_table = pd.DataFrame(columns=["Speed", "Peak 1", "Peak 2", "Peak 3", "Peak 4"])
    ratio_data = []  # To store speed vs ratio (Peak 2 / Peak 1)

    plt.figure(figsize=(10, 6))

    for i in range(4):
        speed = speeds[i]
        x = data.iloc[:, 2 * i].dropna().values  
        y = data.iloc[:, 2 * i + 1].dropna().values  
        row_data = [speed]  
        color = colors[i % len(colors)]  

        # Ensure x and y have the same length
        min_len = min(len(x), len(y))
        x, y = x[:min_len], y[:min_len]

        plt.plot(x, y, label=f'Original Data (Speed {speed} km/s)', marker='o', linestyle='-', color=color)

        peak_1_y = None
        peak_2_y = None

        for j, center in enumerate(energy_centers):
            lower_bound = center - search_range
            upper_bound = center + search_range
            indices_in_range = np.where((x >= lower_bound) & (x <= upper_bound))[0]

            if len(indices_in_range) == 0:
                row_data.append(None)
                continue

            peaks_in_range, _ = find_peaks(y[indices_in_range], prominence=peak_prominence)

            if len(peaks_in_range) == 0:
                row_data.append(None)
                continue

            largest_peak_index_in_range = peaks_in_range[np.argmax(y[indices_in_range][peaks_in_range])]
            largest_peak_index = indices_in_range[largest_peak_index_in_range]

            x_fit_smooth, y_fit_smooth = fit_gaussian_to_peak(x, y, largest_peak_index, num_points, wid_guess)
            plt.plot(x_fit_smooth, y_fit_smooth, label=f'Fitted Gaussian {j+1} (near {center} eV, Speed {speed} km/s)', linestyle='--', color=color)

            max_gaussian_y = np.max(y_fit_smooth)  
            row_data.append(max_gaussian_y)

            if j == 0:
                peak_1_y = max_gaussian_y
            elif j == 1:
                peak_2_y = max_gaussian_y

        results_table.loc[len(results_table)] = row_data

        if peak_1_y is not None and peak_2_y is not None:
            ratio = peak_2_y / peak_1_y
            ratio_data.append((speed, ratio))

    plt.title('Data with Fitted Gaussians for Different Speeds and Energy Intervals')
    plt.xlabel('x (Energy eV)')
    plt.ylabel('y (Intensity)')
    plt.legend()
    plt.grid(True)
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

# Run the function with the updated speed mapping
plot_data_and_fit("Ne10Henew.csv", peak_prominence=0.005)
