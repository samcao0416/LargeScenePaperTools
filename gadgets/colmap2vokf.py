from scipy.spatial.transform import Rotation as R
import numpy as np
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-in", '--input_path', type=str, required=True)
    parser.add_argument('out', '--output_path', type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    in_txt_path = args.in_txt_path
    out_txt_path = args.out_txt_path

    with open(in_txt_path, 'r') as f:
        lines = f.readlines()

        