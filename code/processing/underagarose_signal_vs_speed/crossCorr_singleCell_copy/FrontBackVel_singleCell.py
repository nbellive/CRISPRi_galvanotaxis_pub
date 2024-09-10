import os
import numpy as np
from skimage import io, img_as_float, transform
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy
from scipy import signal
import pandas as pd

# Directory containing all the velocity kymographs
directory_path = '/Volumes/ExpansionHomesA/Amy/Galvanin_data/Kymographs/Velocity'

# Directory to save plots
save_plots_dir = '/Volumes/ExpansionHomesA/Amy/Galvanin_data/Kymographs/Velocity_save'
if not os.path.exists(save_plots_dir):
    os.makedirs(save_plots_dir)

# Set vmax, -vmax for velocity plots
v_max = 3

# Get a list of all image files in the directory
image_files = [f for f in os.listdir(directory_path) if f.endswith('.tif')]

# Define target dimensions
target_height, target_width = 360, 121

# Initialize lists to store max_x and max_y values
max_x_values = []
max_y_values = []

# Iterate through each image
for image_file in image_files:
    # Load the original image
    image_path = os.path.join(directory_path, image_file)
    image = io.imread(image_path)
    image = img_as_float(image)

    image = image[:, 0:120]

    # Multiply each pixel intensity by 12
    image *= 12

    # Resize image to target dimensions
    resized_image = transform.resize(image, (target_height, target_width))

    # Slice the array to keep only the specified x-coordinates
    off_x_value = 0
    on_x_value = 120
    off_on_sig = resized_image[:, off_x_value:on_x_value]

    # Define the range of y-values for the front and back
    # Because the pixels are defined with 0 at top and 360 at bottom (inverse of graphed), use these values
    # Front y-values
    degree_60 = 240
    degree_120 = 300
    # Back y-values
    degree_240 = 60
    degree_300 = 120

    # OFF-ON
    # Slice the array to keep only the specified y-coordinates
    front_off_on_sig = off_on_sig[degree_60:degree_120, :]

    # Back splice
    # Slice the array to keep only the specified y-coordinates
    back_off_on_sig = off_on_sig[degree_240:degree_300, :]

    # Assuming img is your image with shape (60, 120)
    img_front = front_off_on_sig
    img_back = back_off_on_sig

    # Calculate the average intensity along columns
    average_intensity_front = np.mean(img_front, axis=0)
    average_intensity_back = np.mean(img_back, axis=0)

    # Cross-correlation
    def cross_correlation(x, y):
        m = len(x)
        corr = np.correlate(x - x.mean(), y - y.mean(), mode='full') / (m * x.std() * y.std())
        return corr

    corr = cross_correlation(average_intensity_front, average_intensity_back)
    lags = signal.correlation_lags(len(average_intensity_back), len(average_intensity_front))
    lags = lags / 12

    # Find the maximum correlation value and plot the coordinates on the graph
    def find_max_coordinates(x, y):
        max_index = np.argmin(y)  # Find the index of the maximum value in y
        max_x = x[max_index]      # Get the x value corresponding to the maximum y value
        max_y = y[max_index]      # Get the maximum y value
        return max_x, max_y

    max_x, max_y = find_max_coordinates(lags, corr)
    max_x_values.append(max_x)
    max_y_values.append(max_y)

    # Plotting the data and saving the plot
    plt.plot(lags, corr, marker='o', linestyle='', markerfacecolor='mediumvioletred', markeredgewidth=0)
    plt.text(max_x + 0.3, max_y, f'({max_x:.4f}, {max_y:.4f})', fontsize=10, verticalalignment='center')
    plt.xlabel('Lags (min)')
    plt.ylabel('Correlation')
    plt.title('Front Back Vel Correlation vs Lags')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_plots_dir, f'{image_file}_correlation_plot.jpg'))
    plt.close()

# Save max_x and max_y values to CSV
max_values_df = pd.DataFrame({'Image File': image_file, 'Lags 4': max_x_values, 'Correlation 4': max_y_values})
max_values_csv_path = os.path.join(save_plots_dir, 'max_coordinates4.csv')
max_values_df.to_csv(max_values_csv_path, index=False)

# Generate histogram plots for max_x and max_y values
def plot_histogram(data, xlabel, title, save_path):
    plt.figure()
    plt.hist(data, bins=30, alpha=0.75, edgecolor='black')
    median = np.median(data)
    std_dev = np.std(data)
    plt.axvline(median, color='red', linestyle='dashed', linewidth=1)
    plt.axvline(median + std_dev, color='blue', linestyle='dashed', linewidth=1)
    plt.axvline(median - std_dev, color='blue', linestyle='dashed', linewidth=1)
    plt.text(median, plt.ylim()[1]*0.9, f'Median: {median:.2f}', color='red')
    plt.text(median + std_dev, plt.ylim()[1]*0.8, f'+1 SD: {median + std_dev:.2f}', color='blue')
    plt.text(median - std_dev, plt.ylim()[1]*0.8, f'-1 SD: {median - std_dev:.2f}', color='blue')
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.title(title)
    plt.savefig(save_path)
    plt.show()
    plt.close()

# Plot histogram for max_x
plot_histogram(max_x_values, 'Lags', 'Front Back Vel Lags', os.path.join(save_plots_dir, 'histogram_Lags.jpg'))

# Plot histogram for max_y
plot_histogram(max_y_values, 'Correlation Values', 'Front Back Vel Correlation Values', os.path.join(save_plots_dir, 'histogram_correlations.jpg'))
