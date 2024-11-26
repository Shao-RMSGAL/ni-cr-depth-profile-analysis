import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import splrep, splev
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import os
import shutil
import re


# Save data to a csv
def save_data(depth, profile, path):
    data = {
            "Depth (micrometers)": np.linspace(0, depth, len(profile)),
            "Profile": profile,
            }
    df = pd.DataFrame(data)
    try:
        df.to_csv(path, index=False)
        print("Saved", path)
    except Exception as e:
        print("Could not save.", e)


def get_profile(arr_val):
    return np.sum(np.all(arr_val == [0, 0, 0], axis=2), axis=1)


def get_image_depth(image_name):
    # For x3500, the scale bar (10 micrometer)
    #   is 130 pixels wide. The image is 769 pixels tall
    # For x1000, the scale bar (50 micrometer)
    #   is 185 pixels wide. The image is 769 pixels tall
    # For x200, the scale bar (200 micrometer)
    #   is 149 pixels wide. The image is 769 pixels tall
    print("Searching", image_name)
    x = re.search(r"\bx\d+", image_name)
    if x is None:
        return 1
    if x.group() == "x3500":
        return 10 * 769 / 130  # micrometer
    elif x.group() == "x1000":
        return 50 * 769 / 185  # micrometer
    elif x.group() == "x200":
        return 200 * 769 / 149  # micrometer
    else:
        return 1  # Unknown. Default to 1


def create_plot_binary(profile, height, image, path):
    plt.figure(figsize=(3, image.shape[0]/100), dpi=100)
    plt.plot(
            profile,
            np.linspace(0, height, image.shape[0]),
            color='red',
            )
    #  plt.xlabel("Void counts")
    #  plt.ylabel("Depth (μm)")
    plt.gca().invert_yaxis()
    plt.gca().set_ylim(height, 0)
    plt.gca().set_xlim(0, np.amax(profile))
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
    plt.tight_layout(pad=0)
    plt.savefig(path)
    plt.close()


def conduct_analysis(input_path, output_path, threshold):
    print(output_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        print("Made", output_path)
    for ext_dir in os.listdir(input_path):
        total_output_path = os.path.join(output_path, ext_dir)
        if not os.path.exists(total_output_path):
            os.mkdir(total_output_path)
            print("Made", total_output_path)
        subdir = os.path.join(input_path, ext_dir)
        print("For", subdir)
        for img_name in os.listdir(subdir):
            im_path = os.path.join(subdir, img_name)
            image_data = cv.imread(im_path)
            image_data_cropped = image_data[0:-90, :]
            img_no_ext = img_name.split('.')[0]
            depth = get_image_depth(img_no_ext)
            im_save_path = os.path.join(
                    total_output_path,
                    img_no_ext + "_threshold.png")
            profile_save_path = os.path.join(
                    total_output_path,
                    img_no_ext + "_profile.csv")
            plot_save_path = os.path.join(
                    total_output_path,
                    img_no_ext + "_plot.png")

            _, binary = cv.threshold(image_data_cropped,
                                     threshold,
                                     255,
                                     cv.THRESH_BINARY)
            profile = get_profile(binary)
            create_plot_binary(profile, depth, binary, plot_save_path)
            plot_img = cv.imread(plot_save_path)
            os.remove(plot_save_path)
            save_data(depth, profile, profile_save_path)
            image_data = cv.imwrite(
                    im_save_path,
                    np.concatenate(
                        (image_data_cropped, binary, plot_img),
                        axis=1)
                    )
            print("Saved", im_save_path)
        print()
    #  print("Deleting", output_path)
    #  shutil.rmtree(output_path)


def main():
    images_dir = "images"
    files_subdir = "These Images to Find Void Density"
    output_dir_name = "output"
    input_dir = os.path.join(".", images_dir, files_subdir)
    output_dir = os.path.join(".", output_dir_name, files_subdir)
    threshold = 45
    conduct_analysis(input_dir, output_dir, threshold)


if __name__ == "__main__":
    main()
