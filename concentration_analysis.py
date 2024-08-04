import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import splrep, splev
from scipy.ndimage import gaussian_filter1d
import pandas as pd


# Save data to a csv
def save_data(segments, depth, cr_profile, ni_profile):
    data = {
            "Segment": np.arange(1, segments + 1),
            "Depth (nm)": np.linspace(0, depth, segments),
            "Cr precipitate coverage (%)": cr_profile * 100,
            "Ni precipitate coverage (%)": ni_profile * 100
            }
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv("./output/precipitate_coverage.csv", index=False)


# Plot profiles and save a figure
def plot_profiles(cr_profile, ni_profile, height):
    depths = np.linspace(0, height, segments)

    # Optional data smoothing
    sigma = 2.0
    cr_conc_profile_sm = gaussian_filter1d(cr_profile, sigma)
    ni_conc_profile_sm = gaussian_filter1d(ni_profile, sigma)

    # Optional interpolation for smoother graph
    cr_spline = splrep(depths, cr_conc_profile_sm, s=0)
    ni_spline = splrep(depths, ni_conc_profile_sm, s=0)

    # Generate spline for smoother plot
    spline_points = 500
    depths_spline = np.linspace(depths.min(), depths.max(), spline_points)
    cr_spline_data = splev(depths_spline, cr_spline)
    ni_spline_data = splev(depths_spline, ni_spline)

    # Plot Cr data
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)  # type: ignore
    fig.suptitle("Ni and Cr Precipitate Coverage vs Sample Depth")
    ax1.plot(depths_spline, cr_spline_data * 100)
    ax1.set_xlim(0, height)
    ax1.set_ylim(0, 15)
    ax1.set_ylabel("Cr Precipitate coverage (%)")
    ax1.grid()

    # Plot Ni data
    ax2.plot(depths_spline, ni_spline_data * 100)
    ax2.set_xlim(0, height)
    ax2.set_ylim(0, 3)
    ax2.set_xlabel("Depth (nm)")
    ax2.set_ylabel("Ni Precipitate coverage (%)")
    ax2.grid()

    # Save figure
    fig.savefig("./output/Cr_Ni_Precipitate_Coverage_Plot.png")


# Show binary images and save a figure
def show_binary_images(cr_binary, ni_binary, width, height):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)  # type: ignore

    # Plot Cr data
    ax1.imshow(cr_binary, cmap='gray', extent=[0, width, height, 0])
    ax1.set_xlabel("Width (nm)")
    ax1.set_ylabel("Depth (nm)")
    ax1.set_xlim(0, width)
    ax1.set_ylim(height, 0)
    ax1.set_title("Cr Precipitate Coverage")

    # Plot Ni data
    ax2.imshow(ni_binary, cmap='gray', extent=[0, width, height, 0])
    ax2.set_xlabel("Width (nm)")
    ax2.set_xlim(0, width)
    ax2.set_ylim(height, 0)
    ax2.set_title("Ni Precipitate Coverage")

    # Save figure
    fig.savefig("./output/Cr_Ni_Precipitate_Coverage.png")


# Calculate concentration profiles and return them
def get_conc_profiles(cr_binary, ni_binary, segments):
    # Get the number of rows of pixels
    cr_row_count = cr_binary.shape[0]
    ni_row_count = ni_binary.shape[0]
    # Determine the interval of pixels between segments
    cr_interval = cr_row_count // segments
    ni_interval = ni_row_count // segments
    # Preallocate the profiles
    cr_profile = np.zeros(segments)
    ni_profile = np.zeros(segments)
    for i in range(0, segments):
        # Calculate the segment indexes to count pixels
        cr_prev_index = i * cr_interval
        ni_prev_index = i * ni_interval
        cr_next_index = (i + 1) * cr_interval
        ni_next_index = (i+1) * ni_interval
        # Select the segment from the pixel data
        cr_segment = cr_binary[cr_prev_index:cr_next_index, :]
        ni_segment = ni_binary[ni_prev_index:ni_next_index, :]
        # Count white pixels in the segment
        cr_profile[i] = np.sum(cr_segment == 255) / cr_segment.size
        ni_profile[i] = np.sum(ni_segment == 255) / ni_segment.size
    return cr_profile, ni_profile


# Import images
cr_img = cv.imread("./images/1953 45000 x SI EDS-HAADF-Cr-at.bmp")
ni_img = cv.imread("./images/1953 45000 x SI EDS-HAADF-Ni-at.bmp")

# Convert to grayscale (Not strictly needed)
cr_gray = cv.cvtColor(cr_img, cv.COLOR_BGR2GRAY)
ni_gray = cv.cvtColor(ni_img, cv.COLOR_BGR2GRAY)

# Image dimensions nm (Same for both)
width = 900
height = 2000

# Extract binary data using thresholds
cr_threshold = 85
ni_threshold = 85
_, cr_binary = cv.threshold(cr_img, cr_threshold, 255, cv.THRESH_BINARY)
_, ni_binary = cv.threshold(ni_img, ni_threshold, 255, cv.THRESH_BINARY)

# Number of segments to use when profiling
segments = 50
cr_profile, ni_profile = get_conc_profiles(cr_binary, ni_binary, segments)

plot_profiles(cr_profile, ni_profile, height)
show_binary_images(cr_binary, ni_binary, width, height)
save_data(segments, height, cr_profile, ni_profile)
