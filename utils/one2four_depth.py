import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("depths_folder", type=str)
args = parser.parse_args()

depths_folder = args.depths_folder

depth_img_list = sorted(glob(os.path.join(depths_folder, "*.png")))

for depth_img in tqdm(depth_img_list):

    depth_img_single = cv2.imread(depth_img, cv2.IMREAD_GRAYSCALE)

    bgr_depth = cv2.cvtColor(depth_img_single, cv2.COLOR_GRAY2BGR)

    alpha_depth = np.where(depth_img_single > 0, 255, 0)

    bgra_depth = np.dstack((bgr_depth, alpha_depth))

    cv2.imwrite(depth_img, bgra_depth)