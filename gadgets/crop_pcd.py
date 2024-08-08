import argparse
import numpy as np
import os
import open3d as o3d
from tqdm import tqdm

def arg_parser():
    parser = argparse.ArgumentParser(description='Crop pointclouds')
    parser.add_argument('--input_pcd', '-in', type=str, required=True, help='The pointcloud location')
    parser.add_argument('--output_folder', '-out',type=str, required=True, help='The folder to save the cropped pointclouds')
    parser.add_argument('--crop_txt', '-txt', type=str,required=True, help= 'pointcloud crop reference file')
    parser.add_argument('--expand', type=float, default=0.3, help = "expand threshold")
    args = parser.parse_args()
    return args

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

def get_blocks(txt_path, expand_threshold):
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

def _do_crop_with_quad(pcd, quad):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    new_points = []
    new_colors = []

    for index, point in enumerate(tqdm(points)):
        point_xy = point[0:2]
        if is_point_inside_quad(quad, point_xy):
            new_points.append(point)
            try:
                new_colors.append(colors[index])
            except:
                new_colors.append([254,254,254])

    new_points = np.array(new_points)
    new_colors = np.array(new_colors)

    new_ply = o3d.geometry.PointCloud()
    new_ply.points = o3d.utility.Vector3dVector(new_points)
    new_ply.colors = o3d.utility.Vector3dVector(new_colors)

    return new_ply

def cross_product(p1, p2):
    return p1[0] * p2[1] - p1[1] * p2[0]

def is_left_turn(p, q, r):
    return cross_product([q[0]-p[0], q[1]-p[1]], [r[0]-q[0], r[1]-q[1]]) > 0

def is_point_inside_quad(quad, point):
    if len(quad) != 4:
        raise ValueError("The quadrilateral must have exactly 4 points")

    check = is_left_turn(quad[0], quad[1], point)
    for i in range(1, 4):
        if is_left_turn(quad[i], quad[(i+1)%4], point) != check:
            return False
    return True


if __name__ == "__main__":

    args = arg_parser()
    input_pcd = args.input_pcd
    output_folder = args.output_folder
    crop_txt = args.crop_txt
    expand_threshold = args.expand

    quad_list = get_blocks(crop_txt, expand_threshold)

    os.makedirs(output_folder, exist_ok=True)

    pcd = o3d.io.read_point_cloud(input_pcd)

    for index, quad in enumerate(quad_list):

        new_pcd = _do_crop_with_quad(pcd, quad)

        output_path = os.path.join(output_folder, os.path.basename(input_pcd)[:-4] +"_%02d.ply" %(index))

        o3d.io.write_point_cloud(output_path, new_pcd)


    
    