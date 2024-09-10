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

# Directory where generated images are saved
directory_save = ('')

# Signal maximum and minimum intensity
v_max = 0.8
v_min = 0.4

# Velocity maximum (even integer)
vmax= 2

# SIGNAL
# Get a list of all image files in the directory
image_files = [f for f in os.listdir(directory_path) if f.endswith('.tif')]

# Load the first image to get dimensions
first_image_path = os.path.join(directory_path, image_files[0])
first_image = io.imread(first_image_path)
first_image = img_as_float(first_image)

# Define target dimensions
target_height, target_width = 360, 121

# Initialize an array to store normalized pixel intensities
normalized_pixel_intensity_sum = np.zeros((target_height, target_width), dtype=np.float64)

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

    # # Save image for individual cell
    # output_normImage = '/Users/amyplatenkamp/Desktop/0.3mA combined kymograph data/Average SD/pixel_std_deviation_vel.tif'
    # io.imsave(output_path_std_dev, pixel_std_deviation)

    # Add normalized pixel intensities to the sum
    normalized_pixel_intensity_sum += resized_image

# Calculate the average normalized pixel intensity
average_normalized_pixel_intensity = normalized_pixel_intensity_sum / len(image_files)

# Calculate the standard deviation of each pixel across all images
pixel_std_deviation = np.std([transform.resize(img_as_float(io.imread(os.path.join(directory_path, f)))[:, 0:120],
                                                (target_height, target_width))
                              for f in image_files], axis=0, dtype=np.float64)

# # Save the averaged normalized image
# output_path_avg = '/Users/amyplatenkamp/Desktop/1.2mA combined kymograph data/Avg kymographs output/average_normalized_image_sig.tif'
# io.imsave(output_path_avg, average_normalized_pixel_intensity)
#
# # Save the standard deviation image
# output_path_std_dev = '/Users/amyplatenkamp/Desktop/1.2mA combined kymograph data/Avg kymographs output/pixel_std_deviation_sig.tif'
# io.imsave(output_path_std_dev, pixel_std_deviation)

# Create meshgrid for x-coordinates
x_coordinates = np.arange(1, target_width+1)

# Create meshgrid for normalized y-coordinates
normalized_y = np.linspace(0, 360, target_height)[:, np.newaxis]
x_mesh, y_mesh = np.meshgrid(x_coordinates, normalized_y)

# # Assuming image_array is your image array of size (360, 181)
#
# # Take the top 90 rows
# top_90_rows = average_normalized_pixel_intensity[:90, :]
#
# # Take the remaining rows
# bottom_rows = average_normalized_pixel_intensity[90:, :]
#
# # Concatenate the bottom rows followed by the top 90 rows to put them at the bottom
# average_normalized_pixel_intensity = np.concatenate((bottom_rows, top_90_rows), axis=0)
#
# # Verify the new shape of the image array
# print("New shape of the image array:", average_normalized_pixel_intensity.shape)
#
# # Assuming image_array is your image array of size (360, 181)
#
# # Take the top 90 rows
# top_90_rows = pixel_std_deviation[:90, :]
#
# # Take the remaining rows
# bottom_rows = pixel_std_deviation[90:, :]
#
# # Concatenate the bottom rows followed by the top 90 rows to put them at the bottom
# pixel_std_deviation = np.concatenate((bottom_rows, top_90_rows), axis=0)
#
# # Verify the new shape of the image array
# print("New shape of the sd array:", pixel_std_deviation.shape)

# # Save the averaged normalized image
# output_path_avg = '/Users/amyplatenkamp/Desktop/1.2mA combined kymograph data/Avg kymographs output/average_normalized_image_sig_shift.tif'
# io.imsave(output_path_avg, average_normalized_pixel_intensity)
#
# # Save the standard deviation image
# output_path_std_dev = '/Users/amyplatenkamp/Desktop/1.2mA combined kymograph data/Avg kymographs output/pixel_std_deviation_sig_shift.tif'
# io.imsave(output_path_std_dev, pixel_std_deviation)


# Plot the averaged normalized image
plt.figure(figsize=(8, 6))

plt.subplot(1, 2, 1)
plt.imshow(average_normalized_pixel_intensity, vmax = v_max, vmin = v_min, cmap='inferno', aspect='auto', extent=[0, target_width/ 12, 0, 360])
plt.title('Averaged Normalized Image')
plt.yticks(np.arange(0, 361,step=30))
plt.xticks(np.arange(0, 11, step=5))
plt.xlabel('Time (min)')
plt.colorbar()

# Plot the standard deviation of each pixel
plt.subplot(1, 2, 2)
plt.imshow(pixel_std_deviation, cmap='inferno', aspect='auto', extent=[0, target_width/12, 0, 360])
plt.title('Standard Deviation of Each Pixel')
plt.yticks(np.arange(0, 361,step=30))
plt.xticks(np.arange(0, 11, step=5))
plt.xlabel('Time (min)')
plt.colorbar()

plt.tight_layout()
plt.show()

# Plot the averaged normalized image
plt.figure(figsize=(4, 6))
plt.imshow(average_normalized_pixel_intensity, vmax = v_max, vmin = v_min, cmap='inferno', aspect='auto', extent=[0, target_width/ 12, 0, 360])
plt.title('Averaged Normalized Image')
plt.yticks(np.arange(0, 361,step=30))
plt.xticks(np.arange(0, 11, step=5))
plt.xlabel('Time (min)')
plt.colorbar()

# Add transparent boxes
# rect1 = patches.Rectangle((0, 60), 120 / 12, 60, linewidth=3, edgecolor='black', facecolor='none')
rect2 = patches.Rectangle((0, 240), 120 / 12, 60, linewidth=3, edgecolor='black', facecolor='none')
# plt.gca().add_patch(rect1)
plt.gca().add_patch(rect2)

plt.tight_layout()
plt.show()

# Define the range of x-values to keep (60-120)
off_x_value = 0
on_x_value = 120
on2_x_value = 61
off2_x_value = 181

# Slice the array to keep only the specified x-coordinates
off_on_sig = average_normalized_pixel_intensity[:, off_x_value:on_x_value]
off_on_sd = pixel_std_deviation[:, off_x_value:on_x_value]

plt.figure(figsize=(8, 6))

plt.subplot(1, 2, 1)
plt.imshow(off_on_sig, cmap='inferno', aspect='auto', extent=[0, 120/12, 0, 360])
plt.title('Off-On Sig')
plt.xticks(np.arange(0, 11, step=5))
plt.xlabel('Time (min)')
plt.colorbar()

# Plot the standard deviation of each pixel
plt.subplot(1, 2, 2)
plt.imshow(off_on_sd, cmap='inferno', aspect='auto', extent=[0, 120/12, 0, 360])
plt.title('Off-On SD')
plt.xticks(np.arange(0, 11, step=5))
plt.xlabel('Time (min)')
plt.colorbar()

plt.tight_layout()
plt.show()

# Verify the new shape of the image arrays
print("Shape of Off-On:", off_on_sig.shape)
print("Shape of Off-On SD:", off_on_sd.shape)

# Splice to give the values in normalized position to average over

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
back_off_on_sig = off_on_sig[degree_240:degree_300, :]
back_off_on_sd = off_on_sd[degree_240:degree_300, :]

plt.figure(figsize=(8, 2))

plt.subplot(1, 2, 1)
color_backsig = plt.imshow(back_off_on_sig, cmap='inferno', aspect='auto', extent=[0, 120/12, 240, 300])
plt.title('Back Off-On Sig')
plt.xticks(np.arange(0, 11, step=5))
plt.xlabel('Time (min)')
plt.colorbar()

# Plot the standard deviation of each pixel
plt.subplot(1, 2, 2)
plt.imshow(back_off_on_sd, cmap='inferno', aspect='auto', extent=[0, 120/12, 240, 300])
plt.title('Back Off-On SD')
plt.xticks(np.arange(0, 11, step=5))
plt.xlabel('Time (min)')
plt.colorbar()

plt.tight_layout()
plt.show()
# Assuming img is your image with shape (60, 120)
img = back_off_on_sig

# Calculate the average intensity along columns
average_intensity = np.mean(img, axis=0)
print("Shape of average_intensity:", average_intensity.shape)
print(average_intensity)

# Calculate the standard deviation of intensity along columns
std_intensity = np.std(img, axis=0)

print("Average intensity:", average_intensity)
print("Average intensity shape:", average_intensity.shape)

# # Print the average intensity and standard deviation for each column position
# for col, (avg_intensity, std_dev) in enumerate(zip(average_intensity, std_intensity)):
#     print(f"Column {col + 1}: Average Intensity {avg_intensity}, Standard Deviation {std_dev}")

# Create x-axis values
x_back_off_on_sig_avg = np.arange(1, img.shape[1] + 1)/12  # Assuming columns start from 1

print("X axis: ", x_back_off_on_sig_avg)
print("X axis shape: ", x_back_off_on_sig_avg.shape)

# Generate new df
back_off_on_sig_avg = np.column_stack((x_back_off_on_sig_avg, average_intensity))
back_off_on_sig_avg = pd.DataFrame(back_off_on_sig_avg)

print("Shape of back_off_on_sig_avg:", back_off_on_sig_avg.shape)
print(back_off_on_sig_avg)

# Plotting the average values with error bars
plt.errorbar(x_back_off_on_sig_avg, average_intensity, yerr=std_intensity, fmt='o', ecolor='gray', capsize=5, markerfacecolor='mediumvioletred', markeredgewidth=0)
plt.xlabel('Time (min)')
plt.xticks(np.arange(0, 11, step=1))
plt.yticks(np.arange(v_min,v_max +0.1, step=0.1))
plt.ylabel('Average Intensity with Standard Deviation')
plt.title('Off-On Back Intensity')
plt.grid(False)
cbar = plt.colorbar(color_backsig, cmap= 'inferno', cax=plt.gcf().add_axes([0.92, 0.12, 0.02, 0.75]))

# # Set the ticks of the colorbar to match the y-axis values
# cbar.set_ticks(np.linspace(average_intensity.min(), average_intensity.max(), 7))
# cbar.set_ticklabels([f'{val:.2f}' for val in np.linspace(average_intensity.min(), average_intensity.max(), 7)])
cbar.set_ticks([])
cbar.set_ticklabels([])
plt.show()

# # Reflect over x-axis
# average_intensity = - average_intensity
#
# # Plotting the average values with error bars
# plt.errorbar(x_front_off_on_sig_avg, average_intensity, yerr=std_intensity, fmt='o', ecolor='gray', capsize=5, markerfacecolor='mediumvioletred', markeredgewidth=0)
# plt.xlabel('Time (min)')
# plt.xticks(np.arange(0, 11, step=1))
# plt.ylabel('Average Intensity with Standard Deviation')
# plt.title('Off-On Front Intensity, flipped over x-axis')
# plt.grid(False)
# plt.show()

# VELOCITY VELOCITY VELOCITY VELOCITY VELOCITY VELOCITY VELOCITY VELOCITY VELOCITY VELOCITY VELOCITY VELOCITY
# VELOCITY VELOCITY VELOCITY VELOCITY VELOCITY VELOCITY VELOCITY VELOCITY VELOCITY VELOCITY VELOCITY VELOCITY

# Get a list of all image files in the directory
image_files = [f for f in os.listdir(directory_path2) if f.endswith('.tif')]

# Load the first image to get dimensions
first_image_path = os.path.join(directory_path2, image_files[0])
first_image = io.imread(first_image_path)
first_image = img_as_float(first_image)

# Initialize an array to store normalized pixel intensities
normalized_pixel_velocity_sum = np.zeros((target_height, target_width), dtype=np.float64)

# Iterate through each image
for image_file in image_files:
    # Load the original image
    image_path = os.path.join(directory_path2, image_file)
    image = io.imread(image_path)
    image = img_as_float(image)

    # Crop the image to keep columns 0-120 and all rows
    image = image[:, 0:120]

    # Multiply each pixel intensity by 12
    image *= 12

    # Resize image to target dimensions
    resized_image = transform.resize(image, (target_height, target_width))

    # # Save image for individual cell
    # output_normImage = '/Users/amyplatenkamp/Desktop/0.3mA combined kymograph data/Average SD/pixel_std_deviation_vel.tif'
    # io.imsave(output_path_std_dev_vel, pixel_std_deviation_vel)

    # Add pixel intensities to the sum
    normalized_pixel_velocity_sum += resized_image

# Calculate the average pixel velocity
average_normalized_pixel_velocity = normalized_pixel_velocity_sum / len(image_files)

# Calculate the standard deviation of each pixel across all images
pixel_stddev_vel = np.std([transform.resize(img_as_float(io.imread(os.path.join(directory_path2, f)))[:, 0:120],
                                                (target_height, target_width))
                              for f in image_files], axis=0, dtype=np.float64)

# # Save the averaged image
# output_path_avg_vel = '/Users/amyplatenkamp/Desktop/1.2mA combined kymograph data/Avg kymographs output/average_normalized_image_vel.tif'
# io.imsave(output_path_avg_vel, average_normalized_pixel_velocity)
#
# # Save the standard deviation image
# output_path_std_dev_vel = '/Users/amyplatenkamp/Desktop/1.2mA combined kymograph data/Avg kymographs output/pixel_stddev_vel.tif'
# io.imsave(output_path_std_dev_vel, pixel_stddev_vel)

# Create meshgrid for x-coordinates
x_coordinates = np.arange(1, target_width+1)

# Create meshgrid for normalized y-coordinates
normalized_y = np.linspace(0, 360, target_height)[:, np.newaxis]
x_mesh, y_mesh = np.meshgrid(x_coordinates, normalized_y)

# Assuming image_array is your image array of size (360, 181)

# # Take the top 90 rows
# top_90_rows = average_normalized_pixel_velocity[:90, :]
#
# # Take the remaining rows
# bottom_rows = average_normalized_pixel_velocity[90:, :]
#
# # Concatenate the bottom rows followed by the top 90 rows to put them at the bottom
# average_normalized_pixel_velocity = np.concatenate((bottom_rows, top_90_rows), axis=0)
#
# # Verify the new shape of the image array
# print("New shape of the image array:", average_normalized_pixel_velocity.shape)
#
# # Assuming image_array is your image array of size (360, 181)
#
# # Take the top 90 rows
# top_90_rows = pixel_stddev_vel[:90, :]
#
# # Take the remaining rows
# bottom_rows = pixel_stddev_vel[90:, :]
#
# # Concatenate the bottom rows followed by the top 90 rows to put them at the bottom
# pixel_stddev_vel = np.concatenate((bottom_rows, top_90_rows), axis=0)
#
# # Verify the new shape of the image array
# print("New shape of the vel sd array:", pixel_stddev_vel.shape)

# # Save the averaged normalized image
# output_path_avg_vel = '/Users/amyplatenkamp/Desktop/1.2mA combined kymograph data/Avg kymographs output/average_normalized_image_vel_shift.tif'
# io.imsave(output_path_avg_vel, average_normalized_pixel_velocity)
#
# # Save the standard deviation image
# output_path_std_dev_vel = '/Users/amyplatenkamp/Desktop/1.2mA combined kymograph data/Avg kymographs output/pixel_stddev_vel_shift.tif'
# io.imsave(output_path_std_dev_vel, pixel_stddev_vel)


# Plot the averaged normalized image
plt.figure(figsize=(8, 6))

plt.subplot(1, 2, 1)
color_backvel = plt.imshow(average_normalized_pixel_velocity, vmax = vmax, vmin = -vmax, cmap='PiYG', aspect='auto', extent=[0, target_width/ 12, 0, 360])
plt.title('Averaged Velocity')
plt.yticks(np.arange(0, 361, step=30))
plt.xticks(np.arange(0, 11, step=5))
plt.xlabel('Time (min)')
plt.colorbar()

# Plot the standard deviation of each pixel
plt.subplot(1, 2, 2)
plt.imshow(pixel_stddev_vel, cmap='PiYG', aspect='auto', extent=[0, target_width/12, 0, 360])
plt.title('Velocity SD of Each Pixel')
plt.yticks(np.arange(0, 361, step=30))
plt.xticks(np.arange(0, 11, step=5))
plt.xlabel('Time (min)')
plt.colorbar()

plt.tight_layout()
plt.show()

# Slice the array to keep only the specified x-coordinates
off_on_vel = average_normalized_pixel_velocity[:, off_x_value:on_x_value]
off_on_sd_vel = pixel_stddev_vel[:, off_x_value:on_x_value]

plt.figure(figsize=(4, 6))

plt.imshow(off_on_vel, vmax= vmax, vmin = -vmax, cmap='PiYG', aspect='auto', extent=[0, 120/12, 0, 360])
plt.title('Off-On Vel')
plt.yticks(np.arange(0, 361, step=30))
plt.xticks(np.arange(0, 11, step=1))
plt.xlabel('Time (min)')
plt.colorbar()

# Add transparent boxes
# rect1 = patches.Rectangle((0, 60), 120 / 12, 60, linewidth=3, edgecolor='black', facecolor='none')
rect2 = patches.Rectangle((0, 240), 120 / 12, 60, linewidth=3, edgecolor='black', facecolor='none')
# plt.gca().add_patch(rect1)
plt.gca().add_patch(rect2)

plt.tight_layout()
plt.show()

# Verify the new shape of the image arrays
print("Shape of Off-On Vel:", off_on_vel.shape)
print("Shape of Off-On Vel SD:", off_on_sd_vel.shape)

# # Verify the new shape of the image arrays
# print("Shape of On-Off Vel:", on_off_vel.shape)
# print("Shape of On-Off Vel SD:", on_off_sd_vel.shape)

# Splice to give the values in normalized position to average over

# OFF-ON

# Back splice
# Slice the array to keep only the specified y-coordinates
back_off_on_vel = off_on_vel[degree_240:degree_300, :]
back_off_on_sd_vel = off_on_sd_vel[degree_240:degree_300, :]

plt.figure(figsize=(8, 2))

plt.subplot(1, 2, 1)
plt.imshow(back_off_on_vel, vmax = vmax, vmin = -vmax, cmap='PiYG', aspect='auto', extent=[0, 120/12, 240, 300])
plt.title('Back Off-On Vel')
plt.xticks(np.arange(0, 11, step=5))
plt.xlabel('Time (min)')
plt.colorbar()

# Plot the standard deviation of each pixel
plt.subplot(1, 2, 2)
plt.imshow(back_off_on_sd_vel, cmap='PiYG', aspect='auto', extent=[0, 120/12, 240, 300])
plt.title('Back Off-On Vel SD')
plt.xticks(np.arange(0, 11, step=5))
plt.xlabel('Time (min)')
plt.colorbar()

plt.tight_layout()
plt.show()

# Verify the new shape of the image arrays
print("Shape of Back Off-On Vel:", back_off_on_vel.shape)
print(back_off_on_vel.shape)
print("Shape of Back Off-On SD:", back_off_on_sd_vel.shape)

# Assuming img is your image with shape (60, 120)
img2 = back_off_on_vel

# Calculate the average velocity along columns
average_velocity = np.mean(img2, axis=0)
print("Shape of average_velocity:", average_velocity.shape)
print(average_velocity)

# Calculate the standard deviation of velocity along columns
std_velocity = np.std(img2, axis=0)

print("Average velocity:", average_velocity)
print("Average velocity shape:", average_velocity.shape)

# # Print the average velocity and standard deviation for each column position
# for col, (avg_velocity, std_dev) in enumerate(zip(average_velocity, std_velocity)):
#     print(f"Column {col + 1}: Average Velocity {avg_velocity}, Standard Deviation {std_dev}")

# Create x-axis values
x_back_off_on_vel_avg = np.arange(1, img.shape[1] + 1)/12  # Assuming columns start from 1

print("X axis: ", x_back_off_on_vel_avg)
print("X axis shape: ", x_back_off_on_vel_avg.shape)

# Generate new df
back_off_on_vel_avg = np.column_stack((x_back_off_on_vel_avg, average_velocity))
back_off_on_vel_avg = pd.DataFrame(back_off_on_vel_avg)

print("Shape of back_off_on_vel_avg:", back_off_on_vel_avg.shape)
print(back_off_on_vel_avg)

# Plotting the average values with error bars
plt.errorbar(x_back_off_on_vel_avg, average_velocity, yerr=std_velocity, fmt='o', ecolor='gray', capsize=5, markerfacecolor='mediumvioletred', markeredgewidth=0)
plt.xlabel('Time (min)')
plt.xticks(np.arange(0, 11, step=1))
plt.yticks(np.arange(-vmax, vmax+1, step=1))
plt.ylabel('Average Velocity with Standard Deviation')
plt.title('Off-On Back Velocity')
plt.grid(False)
cbar = plt.colorbar(color_backvel, cmap= 'PiYG', cax=plt.gcf().add_axes([0.92, 0.12, 0.02, 0.75]))

# # Set the ticks of the colorbar to match the y-axis values
# cbar.set_ticks(np.linspace(average_intensity.min(), average_intensity.max(), 7))
# cbar.set_ticklabels([f'{val:.2f}' for val in np.linspace(average_intensity.min(), average_intensity.max(), 7)])
cbar.set_ticks([])
cbar.set_ticklabels([])

plt.show()

# # Reflect over x-axis
# average_velocity = - average_velocity

# # get x and y vectors
# x = x_back_off_on_vel_avg
# y = average_velocity
#
# # calculate polynomial
# z = np.polyfit(x, y, 3)
# f = np.poly1d(z)
#
# # calculate new x's and y's
# x_new = np.linspace(x[0], x[-1], 50)
# y_new = f(x_new)
#
# plt.plot(x,y,'o', x_new, y_new)
# plt.xlim([x[0]-1, x[-1] + 1 ])
# plt.show()

# # Generate new df
# fit_vel = np.column_stack((x_new, y_new))
# fit_vel = pd.DataFrame(fit_vel)

# Cross-correlation
def cross_correlation(x, y):
    m = len(x)
    corr = np.correlate(x - x.mean(), y - y.mean(), mode='full') / (m * x.std() * y.std())
    return corr
corr = cross_correlation(average_intensity, average_velocity)
# corr = pd.DataFrame(corr)
lags = signal.correlation_lags(len(average_velocity), len(average_intensity))
lags = lags/12

# Find the maximum correlation value and plot the coordiantes on the graph.
# Change "argmax" to "argmin" if the two arrays are more greatly negatively correlated.
def find_max_coordinates(x, y):
    max_index = np.argmin(y)  # Find the index of the maximum value in y
    max_x = x[max_index]      # Get the x value corresponding to the maximum y value
    max_y = y[max_index]      # Get the maximum y value
    return max_x, max_y
max_x, max_y = find_max_coordinates(lags, corr)
print("Coordinates of maximum y value: ({}, {})".format(max_x, max_y))

# Plotting the data
plt.plot(lags, corr, marker='o', markerfacecolor='mediumvioletred', markeredgewidth=0)

# Annotate the maximum value
plt.text(max_x + 0.3, max_y, f'({max_x:.4f}, {max_y:.4f})',  fontsize=10, verticalalignment='center')

# Adding labels and title
plt.xlabel('Lags (min)')
plt.ylabel('Correlation')
plt.title('Back Sig Vel Correlation vs Lags')

# Display the plot
plt.grid(False)
plt.show()