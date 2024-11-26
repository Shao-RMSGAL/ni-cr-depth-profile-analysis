import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import splrep, splev
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import os


# Save data to a csv
def save_data(segments, depth, cr_profile, ni_profile, mo_profile, fe_profile, path):
    data = {
            "Segment": np.arange(1, segments + 1),
            "Depth (micrometers)": np.linspace(0, depth, segments),
            "Cr Profile": cr_profile * 100,
            "Ni Profile": ni_profile * 100,
            "Mo Profile": mo_profile * 100,
            "Fe Profile": fe_profile * 100,
            }
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(os.path.join(path, "Cr_Ni_Mo_Fe_data.csv"), index=False)


# Plot profiles and save a figure
def plot_profiles(cr_profile, ni_profile, mo_profile, fe_profile, height, path):
    depths = np.linspace(0, height, segments)

    # Optional data smoothing
    sigma = 2.0
    cr_conc_profile_sm = gaussian_filter1d(cr_profile, sigma)
    ni_conc_profile_sm = gaussian_filter1d(ni_profile, sigma)
    mo_conc_profile_sm = gaussian_filter1d(mo_profile, sigma)
    fe_conc_profile_sm = gaussian_filter1d(fe_profile, sigma)

    # Optional interpolation for smoother graph
    cr_spline = splrep(depths, cr_conc_profile_sm, s=0)
    ni_spline = splrep(depths, ni_conc_profile_sm, s=0)
    mo_spline = splrep(depths, mo_conc_profile_sm, s=0)
    fe_spline = splrep(depths, fe_conc_profile_sm, s=0)

    # Generate spline for smoother plot
    spline_points = 500
    depths_spline = np.linspace(depths.min(), depths.max(), spline_points)
    cr_spline_data = splev(depths_spline, cr_spline)
    ni_spline_data = splev(depths_spline, ni_spline)
    mo_spline_data = splev(depths_spline, mo_spline)
    fe_spline_data = splev(depths_spline, fe_spline)

    fig = plt.figure()
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 1])
    ax1 = fig.add_subplot(gs[0, 0], )
    ax3 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax2 = fig.add_subplot(gs[0, 3], sharey=ax1)
    ax4 = fig.add_subplot(gs[0, 4], sharey=ax1)
    #  plt.subplots_adjust(wspace=5)

    # fig.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1,
    #                     wspace=0.0, hspace=0.0)
    # gs.update(wspace=0.2, hspace=0.4)  # More space between the subplots

    fig.set_figheight(5)
    fig.set_figwidth(12)

    fig.suptitle("Ni, Cr, Mo, and Fe Coverage vs. Depth")
    ax1.plot(cr_spline_data, depths_spline)

    #  ax1.set_xlim(0, 17)
    ax1.set_ylim(height, 0)
    ax1.set_xlabel("Cr counts")
    ax1.set_ylabel("Depth (nm)")
    ax1.grid()

    # Plot Ni data
    ax2.plot(ni_spline_data, depths_spline)
    ax2.set_ylim(height, 0)
    #  ax2.set_xlim(0, 3.5)
    ax2.set_xlabel("Ni counts")
    # ax2.set_aspect(aspect_ratio)
    ax2.grid()

    ax3.plot(mo_spline_data, depths_spline)
    #  ax3.set_xlim(0, 17)
    ax3.set_ylim(height, 0)
    ax3.set_xlabel("Mo counts")
    ax3.set_ylabel("Depth (nm)")
    ax3.grid()

    # PlotFei data
    ax4.plot(fe_spline_data, depths_spline)
    ax4.set_ylim(height, 0)
    #  ax4.set_xlim(0, 3.5)
    ax4.set_xlabel("Fe counts")
    # ax2.set_aspect(aspect_ratio)
    ax4.grid()

    # Disable tick marks for all axes
    #  for ax in [ax2, ax4]:
    #      ax.tick_params(length=0, labelbottom=False, labelleft=False,
    #                     labelright=False, labeltop=False)  # Disable tick marks

    # Save figure
    fig.savefig(os.path.join(path, "Cr_Ni_Mo_Fe_Plot.png"))
    plt.close()


# Show binary images and save a figure
def show_binary_images(cr_binary, ni_binary, mo_binary, fe_binary, width, height, path):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True)  # type: ignore

    # Plot Cr data
    ax1.imshow(cr_binary, cmap='gray', extent=[0, width, height, 0])
    ax1.set_xlabel("Width (nm)")
    ax1.set_ylabel("Depth (nm)")
    ax1.set_xlim(0, width)
    ax1.set_ylim(height, 0)
    ax1.set_title("Cr")

    # Plot Ni data
    ax2.imshow(ni_binary, cmap='gray', extent=[0, width, height, 0])
    ax2.set_xlabel("Width (nm)")
    ax2.set_xlim(0, width)
    ax2.set_ylim(height, 0)
    ax2.set_title("Ni")

    # Plot Mo data
    ax3.imshow(mo_binary, cmap='gray', extent=[0, width, height, 0])
    ax3.set_xlabel("Width (nm)")
    ax3.set_xlim(0, width)
    ax3.set_ylim(height, 0)
    ax3.set_title("Mo")

    # Plot  data
    ax4.imshow(fe_binary, cmap='gray', extent=[0, width, height, 0])
    ax4.set_xlabel("Width (nm)")
    ax4.set_xlim(0, width)
    ax4.set_ylim(height, 0)
    ax4.set_title("Fe")

    # Save figure
    fig.savefig(os.path.join(path, "Cr_Ni_Mo_Fe_Binary.png"))
    plt.close()


# Calculate concentration profiles and return them
def get_conc_profiles(cr_binary, ni_binary, mo_binary, fe_binary, segments):
    # Get the number of rows of pixels
    cr_row_count = cr_binary.shape[0]
    ni_row_count = ni_binary.shape[0]
    mo_row_count = mo_binary.shape[0]
    fe_row_count = fe_binary.shape[0]
    # Determine the interval of pixels between segments
    cr_interval = cr_row_count // segments
    ni_interval = ni_row_count // segments
    mo_interval = mo_row_count // segments
    fe_interval = fe_row_count // segments
    # Preallocate the profiles
    cr_profile = np.zeros(segments)
    ni_profile = np.zeros(segments)
    mo_profile = np.zeros(segments)
    fe_profile = np.zeros(segments)
    for i in range(0, segments):
        # Calculate the segment indexes to count pixels
        cr_prev_index = i * cr_interval
        ni_prev_index = i * ni_interval
        mo_prev_index = i * mo_interval
        fe_prev_index = i * fe_interval
        cr_next_index = (i + 1) * cr_interval
        ni_next_index = (i+1) * ni_interval
        mo_next_index = (i + 1) * mo_interval
        fe_next_index = (i+1) * fe_interval
        # Select the segment from the pixel data
        cr_segment = cr_binary[cr_prev_index:cr_next_index, :]
        ni_segment = ni_binary[ni_prev_index:ni_next_index, :]
        mo_segment = mo_binary[mo_prev_index:mo_next_index, :]
        fe_segment = fe_binary[fe_prev_index:fe_next_index, :]
        # Count white pixels in the segment
        cr_profile[i] = np.sum(cr_segment == 255)  # / cr_segment.size
        ni_profile[i] = np.sum(ni_segment == 255)  # / ni_segment.size
        mo_profile[i] = np.sum(mo_segment == 255)  # / mo_segment.size
        fe_profile[i] = np.sum(fe_segment == 255)  # / fe_segment.size
    return cr_profile, ni_profile, mo_profile, fe_profile


def write_images(images, materials, path):
    for idx in range(0, len(materials)):
        cv.imwrite(os.path.join(path, materials[idx]), images[idx])


class_path = ["impure", "pure", "virgin"]
material_path = ["304l", "316l", "316h", "hastelloy_n", "inconel_625", "ni_200"]
extension = ".tiff"
elements = ["cr.tiff", "ni.tiff", "mo.tiff", "fe.tiff"]

for class_name in class_path:
    for material in material_path:

        # Input and output directories
        input_dir = "images"
        output_dir = "output"

        # Image paths
        path = os.path.join(input_dir, class_name, material)

        # Import images
        cr_img = cv.imread(os.path.join(path, elements[0]))
        ni_img = cv.imread(os.path.join(path, elements[1]))
        mo_img = cv.imread(os.path.join(path, elements[2]))
        fe_img = cv.imread(os.path.join(path, elements[3]))
       
        # Crop images
        cr_img = cr_img[21:-21, :]
        ni_img = ni_img[21:-21, :]
        mo_img = mo_img[21:-21, :]
        fe_img = fe_img[21:-21, :]

        # Convert to grayscale (Not strictly needed)
        cr_gray = cv.cvtColor(cr_img, cv.COLOR_BGR2GRAY)
        ni_gray = cv.cvtColor(ni_img, cv.COLOR_BGR2GRAY)
        mo_gray = cv.cvtColor(mo_img, cv.COLOR_BGR2GRAY)
        fe_gray = cv.cvtColor(fe_img, cv.COLOR_BGR2GRAY)

        # Image dimensions nm (Same for both)
        # For Kyle's image set, 0.406 micrometers / pixel
        width = 225
        height = 208

        # Extract binary data using thresholds
        cr_threshold = 85
        ni_threshold = 85
        mo_threshold = 85
        fe_threshold = 85
        _, cr_binary = cv.threshold(cr_img, cr_threshold, 255, cv.THRESH_BINARY)
        _, ni_binary = cv.threshold(ni_img, ni_threshold, 255, cv.THRESH_BINARY)
        _, mo_binary = cv.threshold(mo_img, mo_threshold, 255, cv.THRESH_BINARY)
        _, fe_binary = cv.threshold(fe_img, fe_threshold, 255, cv.THRESH_BINARY)


        # Number of segments to use when profiling
        segments = 100
        cr_profile, ni_profile, mo_profile, fe_profile = get_conc_profiles(
                cr_binary, ni_binary, mo_binary, fe_binary, segments)

        
        path = os.path.join(output_dir, class_name, material)
       
        write_images([cr_img, ni_img, mo_img, fe_img], elements, path)
        plot_profiles(cr_profile, ni_profile, mo_profile, fe_profile, height,
                                                                        path)
        show_binary_images(cr_binary, ni_binary, mo_binary, fe_binary, width, height, path)
        save_data(segments, height, cr_profile, ni_profile, mo_profile, 
                  fe_profile, path)
