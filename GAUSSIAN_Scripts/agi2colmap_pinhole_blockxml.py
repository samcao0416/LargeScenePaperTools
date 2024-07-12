"""
This script is used for converting AGI calibration to colmap format,
and can be directly read by 3D gaussian splatting
Step 1. Check all the files may used
Step 2. Read and write camera intrinsics and write into /sparse/0/cameras.txt
Step 3. Convert AGI extrinsics to quaternions, then convert them into colmap format, save into /sparse/0/images.txt
        At the same time, resize and save all the calibrated images
Step 4. Downsample the pointcloud and save it into point3D.ply
"""


import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
import os
import argparse
import cv2
import open3d as o3d
import sys
from tqdm import tqdm
import warnings
from datetime import datetime
from natsort import natsorted
from glob import glob
import shutil

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-xml", "--document_xml", required=True, type = str, help="path to block exchange xml file")
    parser.add_argument("-img", "--image_path", type = str, required=True , help="path to image folder")
    parser.add_argument("-out", "--output_path", type = str, default=None, help = "path to output colmap files, which can be runed by 3d gaussian")
    parser.add_argument("-ply", "--ply_path", type = str, default=None, help = "path to input pointcloud file")

    parser.add_argument("--resize_img_scale", type = float, default=-1, help = "resize image by scale")
    parser.add_argument("--resize_img_given_width", type = int, default = -1, help = "resize image by a given width, priority lower than the scale option")
    # parser.add_argument("--downsample_pcd", type = float, default= -1, help="downsample the input pointcloud by scale")

    args = parser.parse_args()
    return args

def load_xml_transform(node):
    pose = np.eye(4)
    rotation_node = node.find('Rotation')
    translation_node = node.find('Center')
    for i in range(3):
        for j in range(3):
            index = "M_" + str(i) + str(j)
            pose[j, i] = float(rotation_node.find(index).text)
    pose[0, 3] = float(translation_node.find('x').text)
    pose[1, 3] = float(translation_node.find('y').text)
    pose[2, 3] = float(translation_node.find('z').text)
    return pose

def convert_save_poses(poses):
    colmap_poses = []
    for pose in poses:
        tx, ty, tz, qx, qy, qz, qw = map(float, pose)
        quat = Rotation.from_quat([qx, qy, qz,qw])
        rot_mat = quat.as_matrix()
        
        transformation = np.eye(4)
        transformation[:3, :3] = rot_mat

        translation = np.array([tx, ty, tz])
        transformation[:3, 3] = translation
        transformation = np.linalg.inv(transformation)

        rot_mat = transformation[:3,:3]
        tx = transformation[0,3]
        ty = transformation[1,3]
        tz = transformation[2,3]
        quat_2 = Rotation.from_matrix(rot_mat)
        qx = quat_2.as_quat()[0]
        qy = quat_2.as_quat()[1]
        qz = quat_2.as_quat()[2]
        qw = quat_2.as_quat()[3]
        colmap_poses.append(np.array([qw, qx, qy, qz, tx, ty, tz]))
    return colmap_poses


class colmap_gaussian:
    def __init__(self, args):
        self.args = args
        self.xml_path = args.document_xml
        self.img_path = args.image_path
        self.ply_path = args.ply_path
        self.output_path = args.output_path
        """
        output_folder structure:
        output_folder:
            --images
                --image0
                --image1
                ...
            --sparse
                --0
                    --cameras.txt (intrinsic)
                    --images.txt (extrinsic) Note: for linux, the extension names are case-sentitve
                    --points3D.ply
        """

    def check_files(self):
        # check xml
        print("Checking essential files")
        try:
            xml_tree = ET.parse(self.xml_path)
            print("XML Ok.")
            self.photo_xml = xml_tree.getroot().find("Block").find("Photogroups").find("Photogroup")
        except:
            raise FileNotFoundError("XML doesn't exists")

        # check ply
        try:
            self.pcd = o3d.io.read_point_cloud(self.ply_path)
            if self.pcd is not None:
                print("Ply Ok.")
            else:
                warnings.warn("Ply file doesn't exists, pointcloud process will be neglected, this might cause further issues.", RuntimeError)
                self.pcd = None
        except:
            warnings.warn("Ply file doesn't exists, pointcloud process will be neglected, this might cause further issues.", RuntimeError)
            self.pcd = None
        # self.images = []
        # self.image_names = []
        print("Checking images")

        # Find all images in listed extenstions
        if sys.platform == "win32":
            img_exts = ['png', 'jpg', 'jpeg']
        elif sys.platform == "linux":
            img_exts = ['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG']
        
        self.image_paths = []
        for ext in img_exts:
            self.image_paths += natsorted(glob(os.path.join(self.img_path, "*." + ext)))
        # for img_path in tqdm(image_paths):
        #     image = cv2.imread(img_path)
        #     self.images.append(image)
            #self.image_names.append(os.path.basename(img_path))

        if len(self.image_paths) == 0:
            raise FileNotFoundError("No Image Found")
        else:
            print("Images ok")
            self.raw_height, self.raw_width, _ = cv2.imread(self.image_paths[0]).shape

        print("Making Folders")
        if self.output_path == None:
            self.output_path = "colmap_out_" + str("%02d" %datetime.now().month) + str("%02d" %datetime.now().day) + "_" + str("%02d" %datetime.hour) + str("%02d" %datetime.minute)
        os.makedirs(self.output_path, exist_ok=True)
        
        self.outimage_folder = os.path.join(self.output_path, "images")
        self.outother_folder = os.path.join(self.output_path, "sparse", "0")

        os.makedirs(self.outimage_folder, exist_ok=True)
        os.makedirs(self.outother_folder, exist_ok=True)

        self.intrinsic_path = os.path.join(self.outother_folder, "cameras.txt")
        self.extrinsic_path = os.path.join(self.outother_folder, "images.txt")
        self.pointcloud_path = os.path.join(self.outother_folder, "points3D.ply")

    def show_rescale_info(self):
        if self.args.resize_img_scale > 0:
            resize_scale = self.args.resize_img_scale
            self.new_height = round(self.raw_height * resize_scale)
            self.new_width = round(self.raw_width * resize_scale)
            print("Use resize scale %f to resize images" %(resize_scale))
            self.resize_bool = True
            
        elif self.args.resize_img_given_width > 0:
            new_width = self.args.reszie_img_given_width
            self.new_height = round(self.raw_width / new_width * self.raw_height)
            self.new_width  = round(new_width)
            print("Use given width %d to resize images" %self.new_width)
            self.resize_bool = True

        else:
            self.new_height = self.raw_height
            self.new_width = self.raw_width
            self.resize_bool = False

    # def process_img(self, image_path):
        
    #     # image = cv2.imread(image_path)
    #     out_path = os.path.join(self.outimage_folder, os.path.basename(image_path))
    #     # if self.resize_bool:
    #     #     new_image = cv2.resize(image, (self.new_width, self.new_height))
    #     #     cv2.imwrite(out_path, new_image)
    #     # else:
    #     #     cv2.imwrite(out_path, image)
    #     # copy out_path from image_path by hongyu
    #     shutil.copy(image_path, out_path)
    def process_img(self, image_path):
    
        image = cv2.imread(image_path)
        out_path = os.path.join(self.outimage_folder, os.path.basename(image_path))
        if self.resize_bool:
            new_image = cv2.resize(image, (self.new_width, self.new_height))
            cv2.imwrite(out_path, new_image)
        else:
            cv2.imwrite(out_path, image)
    
    def save_intrinsic(self):
        self.show_rescale_info()
        try:
            focal = float(self.photo_xml.find("FocalLengthPixels").text)
            focal = focal * self.new_width / self.raw_width
        except:
            focal = input("Please input focal length: ")
            
        try:
            cx = (float(self.photo_xml.find("PrincipalPoint").find("x").text)) * self.new_width / self.raw_width
            cy = (float(self.photo_xml.find("PrincipalPoint").find("y").text)) * self.new_height / self.raw_height
        except:
            cx = 0.5 * self.new_width
            cy = 0.5 * self.new_height        
        colmap_intrinsic_text = ["1", "PINHOLE", self.new_width, self.new_height, focal, focal, cx, cy]
        with open(self.intrinsic_path, "w+") as file:
            for i in range(len(colmap_intrinsic_text)):
                try:
                    file.write(colmap_intrinsic_text[i])
                except:
                    file.write(str(colmap_intrinsic_text[i]))
                file.write(" ")
            file.write("\n")

    def save_extrinsic_image(self):
        ## Saving AGI Format Poses
        self.agi_poses_quaternion_dict = {"poses":[], "image_names": []}

        img_basename_list = []
        for img_path in self.image_paths:
            img_basename_list.append(os.path.basename(img_path))

        _default_matrix = np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])

        # print(len(self.photo_xml))
        # print(len(self.image_names))
        idx = 0
        for ele in tqdm(self.photo_xml):
            name = ele.tag
            if name == "Photo":
                pose = load_xml_transform(ele.find('Pose'))
                img_name = os.path.basename(ele.find("ImagePath").text)


                if img_name not in img_basename_list:
                    continue
            
                if (pose == _default_matrix).all():
                    continue

                r = Rotation.from_matrix(pose[:3, :3])
                q = r.as_quat()

                image_path = ele.find("ImagePath").text
            
                # print(idx)
                self.agi_poses_quaternion_dict["poses"].append(np.array([pose[0, 3], pose[1, 3], pose[2, 3], q[0], q[1], q[2], q[3]]))
                self.agi_poses_quaternion_dict["image_names"].append(img_name)

                image_path = os.path.join(os.path.dirname(self.image_paths[0]), img_name)
                self.process_img(image_path)

                idx += 1
        
        colmap_poses = convert_save_poses(self.agi_poses_quaternion_dict["poses"]) # in format of [qw, qx, qy, qz, tx, ty, tz]
        # print(colmap_poses)
        # exit()
        with open(self.extrinsic_path, "w+") as file:
            for idx,pose in enumerate(colmap_poses):
                image_name = self.agi_poses_quaternion_dict["image_names"][idx]
                output_line = f"{idx +1 } {pose[0]} {pose[1]} {pose[2]} {pose[3]} {pose[4]} {pose[5]} {pose[6]} 1 {image_name}\n No Content \n"
                file.write(output_line)      


    def save_pointcloud(self):
        # downsample_scale = self.args.downsample_pcd
        # if downsample_scale > 0 and downsample_scale < 1:
        #     print("Downsample pointcloud with scale %f" %downsample_scale)
        #     downsample_k = round(1 / downsample_scale)
        #     new_pcd = self.pcd.uniform_down_sample(every_k_points = downsample_k)
        #     o3d.io.write_point_cloud(self.pointcloud_path, new_pcd)
        # else:
        #     print("Will not downsample pointcloud")
        #     o3d.io.write_point_cloud(self.pointcloud_path, self.pcd)
            
        target_pcd = 7000000
        # calculate point clouds number
        pointcloud_data = o3d.io.read_point_cloud(self.ply_path)
        cur_pcd = len(pointcloud_data.points)
        print("Current pointcloud has %d points" %cur_pcd)
        if cur_pcd > target_pcd:
            downsample_scale = target_pcd / cur_pcd
            print("Downsample pointcloud with scale %f" %downsample_scale)
            downsample_k = round(cur_pcd / target_pcd)
            new_pcd = pointcloud_data.uniform_down_sample(every_k_points = downsample_k)
            o3d.io.write_point_cloud(os.path.join(args.output_path, "sparse", "0", "points3D.ply"), new_pcd)
            print("Saved downsampled pointcloud")
        else:
            print("Will not downsample pointcloud")
            o3d.io.write_point_cloud(os.path.join(args.output_path, "sparse", "0", "points3D.ply"), pointcloud_data)

    
    ### The following functions are underconstruction
    def compare_cameras(self, vokf_path):
        with open(vokf_path, "r") as file:
            vokf_lines = file.readlines()
        
        self.raw_poses = []

        for line in vokf_lines:
            data = line.strip().split()
            tx, ty, tz, qx, qy, qz, qw = map(float, data[1:])
            quat = Rotation.from_quat([qx, qy, qz,qw])
            rot_mat = quat.as_matrix()

            transformation = np.eye(4)
            transformation[:3, :3] = rot_mat

            translation = np.array([tx, ty, tz])
            transformation[:3, 3] = translation

            self.raw_poses.append(transformation)

if __name__ == "__main__":    
    args = parse_args()
    colmap_files = colmap_gaussian(args)
    # Step 1. Check all the files may used
    colmap_files.check_files()
    # Step 2. Read and write camera intrinsics and write into /sparse/0/cameras.txt
    colmap_files.save_intrinsic()
    # Step 3. Convert AGI extrinsics to quaternions, then convert them into colmap format, save into /sparse/0/images.txt
    #         At the same time, resize and save all the calibrated images
    colmap_files.save_extrinsic_image()
    # Step 4. Downsample the pointcloud and save it into point3D.ply
    # colmap_files.save_pointcloud()
    