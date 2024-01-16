import sys
sys.path.append("./")
sys.path.append("../")
sys.path.append("../..")

import os
import argparse
import open3d as o3d
import pymeshlab as ml

from utils.mesh_proj import MeshImageFilter

def parse():

    parser = argparse.ArgumentParser()
    parser.add_argument("-mesh", "--mesh_path", type=str, required=True, help = "Path of original mesh without texture")
    parser.add_argument("-txt", "--txt_path", type=str, required=True, help = "Path of point picking list")
    parser.add_argument("-out", "--output_folder", type=str, default=None, help = "Path of output folders")
    parser.add_argument("-num", "--target_number", type = int, default=5000000, help = "Target number of faces")

    args = parser.parse_args()

    return args

def get_blocks(txt_path, expand_threshold = 0.3):
    if (type(expand_threshold) is not float) and (type(expand_threshold) is not int):
                print("Wrong Expand Type. Should be Float")
                print(expand_threshold, type(expand_threshold))
                return 0 

    with open(txt_path, "r") as f:
        points_xy = []

        while(True):

            line = f.readline()

            if not line:
                break

            ele = line.split(",")

            points_xy.append([float(ele[1]), float(ele[2])])

    quad_list = []

    for i in range(len(points_xy) // 4):
        quad = [points_xy[i*4],
                points_xy[i*4+1],
                points_xy[i*4+2],
                points_xy[i*4+3]]
        
        quad  = _expand_quad(quad, expand_threshold)

        quad_list.append(quad)

    return quad_list

def _expand_quad(quad, expand_threshold):
    center_x = sum(q[0] for q in quad) / 4
    center_y = sum(q[1] for q in quad) / 4
    
    factor = 1 + expand_threshold
    expanded_quad = []
    for q in quad:
        new_x = center_x + factor * (q[0] - center_x)
        new_y = center_y + factor * (q[1] - center_y)
        expanded_quad.append((new_x, new_y))
    
    return expanded_quad

if __name__ == "__main__":
    args = parse()

    mesh_path = args.mesh_path
    txt_path  = args.txt_path
    if args.output_folder is None:
        output_folder = os.path.join(os.path.dirname(txt_path), "cropped_meshes")
    else:
        output_folder = args.output_folder

    target_num = args.target_number

    print(output_folder)

    os.makedirs(output_folder, exist_ok=True)

    quad_list = get_blocks(txt_path)

    mesh = MeshImageFilter(mesh_path = mesh_path, simplify=False)

    for index, quad in enumerate(quad_list):
        print("Processing block", index)
        mesh.meshprocess.crop_mesh(quad=quad)
        cropped_mesh_path = os.path.join(output_folder, f"mesh_{index}.ply")
        mesh.meshprocess.save_cropped_mesh(cropped_mesh_path)

        ms = ml.MeshSet()
        ms.load_new_mesh(cropped_mesh_path)
        ms.apply_filter("meshing_decimation_quadric_edge_collapse", targetfacenum=target_num)

        ms.save_current_mesh(cropped_mesh_path)

    print("Done")