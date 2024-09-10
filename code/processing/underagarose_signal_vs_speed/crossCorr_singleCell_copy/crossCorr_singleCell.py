import os
import numpy as np
from skimage import io, img_as_float, transform
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import signal
import pandas as pd

# Directory containing all the images
directory_path = ('/Volumes/ExpansionHomesA/Amy/Galvanin_data/Kymographs/Signals')

# Directory to save plots
save_plot_directory = ('/Volumes/ExpansionHomesA/Amy/Galvanin_data/Kymographs/Signals_save')


def find_max_coordinates(x, y):
    max_index = np.argmin(y)
    max_x = x[max_index]
    max_y = y[max_index]
    return max_x, max_y

def cross_correlation(x, y):
    m = len(x)
    corr = np.correlate(x - x.mean(), y - y.mean(), mode='full') / (m * x.std() * y.std())
    return corr

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

# Get a list of all image files in the directory
image_files = [f for f in os.listdir(directory_path) if f.endswith('.tif')]

# Define target dimensions
target_height, target_width = 360, 121

# List to store the max coordinates
max_coordinates = []

# Iterate through each image
for image_file in image_files:
    # Load the original image
    image_path = os.path.join(directory_path, image_file)
    image = io.imread(image_path)
    image = img_as_float(image)

    # Crop the image to keep columns 0-120 and all rows
    image = image[:, 0:120]

    # Normalize intensities
    normalized_image = image / image.max()

    # Resize image to target dimensions
    resized_image = transform.resize(normalized_image, (target_height, target_width))

    # Create meshgrid for x-coordinates
    x_coordinates = np.arange(1, target_width + 1)

    # Create meshgrid for normalized y-coordinates
    normalized_y = np.linspace(0, 360, target_height)[:, np.newaxis]
    x_mesh, y_mesh = np.meshgrid(x_coordinates, normalized_y)

    # Define the range of y-values for the front and back
    degree_60 = 240
    degree_120 = 300
    degree_240 = 60
    degree_300 = 120

    # Slice the array to keep only the specified y-coordinates
    front_off_on_sig = resized_image[degree_60:degree_120, :]
    back_off_on_sig = resized_image[degree_240:degree_300, :]

    # Calculate the average intensity along columns
    average_intensity_front = np.mean(front_off_on_sig, axis=0)
    std_intensity_front = np.std(front_off_on_sig, axis=0)

    average_intensity_back = np.mean(back_off_on_sig, axis=0)
    std_intensity_back = np.std(back_off_on_sig, axis=0)

    # Create x-axis values
    x_front_off_on_sig_avg = np.arange(1, front_off_on_sig.shape[1] + 1) / 12
    x_back_off_on_sig_avg = np.arange(1, back_off_on_sig.shape[1] + 1) / 12

    # Generate new df
    front_off_on_sig_avg = pd.DataFrame(np.column_stack((x_front_off_on_sig_avg, average_intensity_front)))
    back_off_on_sig_avg = pd.DataFrame(np.column_stack((x_back_off_on_sig_avg, average_intensity_back)))

    # Cross-correlation
    corr = cross_correlation(average_intensity_front, average_intensity_back)
    lags = signal.correlation_lags(len(average_intensity_back), len(average_intensity_front))
    lags = lags / 12

    # Find the maximum correlation value and plot the coordinates on the graph.
    max_x, max_y = find_max_coordinates(lags, corr)
    max_coordinates.append([image_file, max_x, max_y])

    print(f'Coordinates of maximum y value for {image_file}: ({max_x:.4f}, {max_y:.4f})')

    # Plotting the data
    plt.plot(lags, corr, marker='o', markerfacecolor='mediumvioletred', markeredgewidth=0)

    # Annotate the maximum value
    plt.text(max_x + 0.3, max_y, f'({max_x:.4f}, {max_y:.4f})', fontsize=10, verticalalignment='center')

    # Adding labels and title
    plt.xlabel('Lags (min)')
    plt.ylabel('Correlation')
    plt.title(f'Front Back Sig Correlation vs Lags: {image_file}')
    plt.grid(False)

    # Save the plot to the specified directory
    plot_save_path = os.path.join(save_plot_directory, f'{os.path.splitext(image_file)[0]}_correlation_plot.jpg')
    plt.savefig(plot_save_path)
    plt.close()

# Save the max coordinates to a CSV file
max_coordinates_df = pd.DataFrame(max_coordinates, columns=['Image File', 'Lags 1', 'Correlation 1'])
max_coordinates_df.to_csv(os.path.join(save_plot_directory, 'max_coordinates_1.csv'), index=False)

# Generate and save histogram plots for max_x and max_y
max_x_values = max_coordinates_df['Lags 1']
max_y_values = max_coordinates_df['Correlation 1']


# Plot histogram for max_x
plot_histogram(max_x_values, 'Lags', 'Front Back Sig Lags', os.path.join(save_plot_directory, 'histogram_lags.jpg'))

# Plot histogram for max_y
plot_histogram(max_y_values, 'Correlation Values', 'Front Back Sig Correlation Values', os.path.join(save_plot_directory, 'histogram_correlations.jpg'))

