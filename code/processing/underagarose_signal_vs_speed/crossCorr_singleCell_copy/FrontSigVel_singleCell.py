import os
import numpy as np
from skimage import io, img_as_float, transform
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy
from scipy import signal
import pandas as pd

# DIRECTORIES
# Directory containing all of the signal kymographs
directory_path = '/Volumes/ExpansionHomesA/Amy/Galvanin_data/Kymographs/Signals'

# Directory containing all the velocity kymographs
directory_path2 = '/Volumes/ExpansionHomesA/Amy/Galvanin_data/Kymographs/Velocity'

# Directory to save plots
save_plot_directory = ('/Volumes/ExpansionHomesA/Amy/Galvanin_data/Kymographs/Front_sig_vel_save')

# Get a list of all image files in the directories
image_files_sig = [f for f in os.listdir(directory_path) if f.endswith('.tif')]
image_files_vel = [f for f in os.listdir(directory_path2) if f.endswith('.tif')]

# Define target dimensions
target_height, target_width = 360, 121

# List to store the max coordinates
max_coordinates = []

# Define the range of x-values to keep (60-120)
off_x_value = 0
on_x_value = 120
on2_x_value = 61
off2_x_value = 181

# Front y-values
degree_60 = 240
degree_120 = 300
# Back y-values
degree_240 = 60
degree_300 = 120


# Function to process an image
def process_image(image_path, crop_cols, target_height, target_width):
    image = io.imread(image_path)
    image = img_as_float(image)
    image = image[:, :crop_cols]  # Crop the image to keep specified columns and all rows
    resized_image = transform.resize(image, (target_height, target_width))
    return resized_image

def cross_correlation(x, y):
    m = len(x)
    corr = np.correlate(x - x.mean(), y - y.mean(), mode='full') / (m * x.std() * y.std())
    return corr

# Find the maximum correlation value and plot the coordiantes on the graph.
# Change "argmax" to "argmin" if the two arrays are negatively correlated.
def find_max_coordinates(x, y):
    max_index = np.argmin(y)  # Find the index of the maximum value in y
    max_x = x[max_index]  # Get the x value corresponding to the maximum y value
    max_y = y[max_index]  # Get the maximum y value
    return max_x, max_y

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

# Iterate through each signal image
for image_file_sig in image_files_sig:
    # Extract the common part of the file name
    common_name = image_file_sig.rsplit('_', 1)[0]

    # Find the corresponding velocity image
    image_file_vel = next((f for f in image_files_vel if f.startswith(common_name)), None)

    if image_file_vel:
        # SIGNAL SIGNAL SIGNAL SIGNAL SIGNAL SIGNAL SIGNAL SIGNAL SIGNAL
        # Process signal image
        image_path_sig = os.path.join(directory_path, image_file_sig)
        sig_image = process_image(image_path_sig, 120, target_height, target_width)

        # Slice the array to keep only the specified x- and y-coordinates
        off_on_sig = sig_image[:, off_x_value:on_x_value]
        front_off_on_sig = off_on_sig[degree_60:degree_120, :]
        img = front_off_on_sig
        average_intensity = np.mean(img, axis=0)

        # VELOCITY VELOCITY VELOCITY VELOCITY VELOCITY VELOCITY

        # Process velocity image
        image_path_vel = os.path.join(directory_path2, image_file_vel)
        vel_image = process_image(image_path_vel, 120, target_height, target_width) * 12

        # Slice the array to keep only the specified x-coordinates
        off_on_vel = vel_image[:, off_x_value:on_x_value]
        # Slice the array to keep only the specified y-coordinates
        front_off_on_vel = off_on_vel[degree_60:degree_120, :]
        # Assuming img is your image with shape (60, 120)
        img2 = front_off_on_vel
        # Calculate the average velocity along columns
        average_velocity = np.mean(img2, axis=0)

        # Calculate correlation
        corr = cross_correlation(average_intensity, average_velocity)
        lags = signal.correlation_lags(len(average_velocity), len(average_intensity))
        lags = lags / 12

        max_x, max_y = find_max_coordinates(lags, corr)
        max_coordinates.append([image_file_sig, image_file_vel, max_x, max_y])

        # Plot the correlation
        plt.plot(lags, corr, marker='o', markerfacecolor='mediumvioletred', markeredgewidth=0)

        # Annotate the maximum value
        plt.text(max_x + 0.3, max_y, f'({max_x:.4f}, {max_y:.4f})', fontsize=10, verticalalignment='center')

        # Adding labels and title
        plt.xlabel('Lags (min)')
        plt.ylabel('Correlation')
        plt.title(f'Front Sig Vel Correlation vs Lags: {image_file_sig}')

        # Display the plot
        plt.grid(False)


        # Save the plot to the specified directory
        plot_save_path = os.path.join(save_plot_directory, f'{os.path.splitext(image_file_sig)[0]}_correlation_plot.jpg')
        plt.savefig(plot_save_path)
        plt.show()
        plt.close()

# Save the max coordinates to a CSV file
max_coordinates_df = pd.DataFrame(max_coordinates, columns=['Image File Sig', 'Image File Vel', 'Lags 3', 'Correlation 3'])
max_coordinates_df.to_csv(os.path.join(save_plot_directory, 'max_coordinates3.csv'), index=False)

# Generate and save histogram plots for max_x and max_y
max_x_values = max_coordinates_df['Lags 3']
max_y_values = max_coordinates_df['Correlation 3']

# Plot histogram for max_x
plot_histogram(max_x_values, 'Lags', 'Front Sig Vel Lags', os.path.join(save_plot_directory, 'histogram_lags.jpg'))

# Plot histogram for max_y
plot_histogram(max_y_values, 'Correlation', 'Front Sig Vel Correlation Values', os.path.join(save_plot_directory, 'histogram_correlations.jpg'))
