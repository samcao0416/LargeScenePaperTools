"""
This script is used for converting AGI calibration to colmap format,
and can be directly read by 3D gaussian splatting
Step 1. Check all the files may used
Step 2. Read and write camera intrinsics and write into /sparse/0/cameras.txt
Step 3. Convert AGI extrinsics to quaternions, then convert them into colmap format, save into /sparse/0/images.txt
        At the same time, resize, recenter and save all the calibrated images
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
from typing import NamedTuple
from utils.undistort_fisheye import undistort_cameras_opencv

class IndexRange(NamedTuple):
    start : int
    end : int
    step: int = 1

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


class agi2colmap_insta():
    def __init__(self, 
                 xml_path: str, img_folder: str, ply_path: str, output_path: str, 
                 resize_img_scale : float = -1, 
                 resize_img_given_width : int = -1, 
                 recenter_img: bool = False, 
                 downsample_pcd: float = -1,
                 img_select_range: list = None,
                 bbox: list = None):
        self.xml_path = xml_path
        self.img_path = img_folder
        self.ply_path = ply_path
        self.output_path = output_path

        self.resize_img_scale = resize_img_scale
        self.resize_img_given_width = resize_img_given_width
        self.recenter_img = recenter_img
        self.downsample_pcd = downsample_pcd
        self.img_select_range = img_select_range
        self.bbox = bbox
    
    def check_files(self):
        # check xml
        print("Checking essential files")
        try:
            xml_tree = ET.parse(self.xml_path)
            print("XML Ok.")
            self.photo_xml = xml_tree.getroot().find("Block").find("Photogroups")#.find("Photogroup")
        except:
            raise FileNotFoundError("XML doesn't exists")
        
        self.disposal_xml()

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
        # check images
        if not os.path.exists(self.img_path):
            raise FileNotFoundError("Image Folder Not Found")
        # self.images = []
        # self.image_names = []
        print("Checking images")

        # Find all images in listed extenstions
        if sys.platform == "win32":
            img_exts = ['png', 'jpg', 'jpeg']
        elif sys.platform == "linux":
            img_exts = ['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG']
        
        self.image_paths = []
        
        if self.img_select_range is None:
            for ext in img_exts:
                self.image_paths += natsorted(glob(os.path.join(self.img_path, "*." + ext)))

        elif self.img_select_range.start >= self.img_select_range.end:
            raise ValueError("img selection range error: starting index >= ending index")
        
        else:
            for ext in img_exts:
                self.image_paths += natsorted(glob(os.path.join(self.img_path, "*." + ext)))[self.img_select_range.start:(self.img_select_range.end + 1):self.img_select_range.step]
        
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
        if self.resize_img_scale > 0:   # under construction
            resize_scale = self.args.resize_img_scale
            self.new_height = round(self.raw_height * resize_scale)
            self.new_width = round(self.raw_width * resize_scale)
            print("Use resize scale %f to resize images" %(resize_scale))
            self.resize_bool = True
            
        elif self.resize_img_given_width > 0:     # under construction
            new_width = self.args.resize_img_given_width
            self.new_height = round(new_width / self.raw_width * self.raw_height)
            self.new_width  = round(new_width)
            print("Use given width %d to resize images" %self.new_width)
            self.resize_bool = True

        else:
            self.new_height = self.raw_height
            self.new_width = self.raw_width
            self.resize_bool = False

        if self.recenter_img:
            self.recenter_bool = True
        else:
            self.recenter_bool = False
    
    def process_img(self, image_path):
        
        image = cv2.imread(image_path)
        out_path = os.path.join(self.outimage_folder, os.path.basename(image_path))
        if self.resize_bool:
            new_image = cv2.resize(image, (self.new_width, self.new_height))
            #cv2.imwrite(out_path, new_image)
        else:
            new_image = image
            #cv2.imwrite(out_path, image)

        if self.recenter_bool:
            final_img = cv2.warpAffine(new_image, self.warp_mat, (self.new_width, self.new_height))
        else:
            final_img = new_image
        
        cv2.imwrite(out_path, final_img)

    def undistort_fisheye(self, xml_path, new_width = 2000, new_height = 2000, fovX = 0.8):
        
        undistorting_cameras = undistort_cameras_opencv(image_path_list=self.image_paths, writing_mode=False)
        undistorting_cameras.read_xml(xml_path=xml_path)
        undistorted_images = undistorting_cameras.set_image_params_and_run(width=new_width, height=new_height, fovX = fovX)
        fx, fy = undistorting_cameras.get_focal()
        cx, cy = undistorting_cameras.get_cxcy()
        self.new_width, self.new_height = undistorting_cameras.get_shape()
        colmap_intrinsic_text = ["1", "PINHOLE", self.new_width, self.new_height, fx, fy, cx, cy]
        with open(self.intrinsic_path, "w+") as file:
            for i in range(len(colmap_intrinsic_text)):
                try:
                    file.write(colmap_intrinsic_text[i])
                except:
                    file.write(str(colmap_intrinsic_text[i]))
                file.write(" ")
            file.write("\n")

        for i in range(len(self.image_paths)):
            image_path = os.path.join(self.outimage_folder, os.path.basename(self.image_paths[i]))
            cv2.imwrite(image_path, undistorted_images[i])

    def save_intrinsic(self):
        self.show_rescale_info()
        
        try:
            focal = float(self.photo_xml.find("FocalLengthPixels").text)
            focal = focal * self.new_width / self.raw_width

            if self.recenter_bool:
                cx = float(self.new_width / 2)
                cy = float(self.new_height / 2)
            else:
                cx = float(self.photo_xml.find("PrincipalPoint")[0].text) * self.new_width / self.raw_width
                cy = float(self.photo_xml.find("PrincipalPoint")[1].text) * self.new_height / self.raw_height
        except:
            focal = input("Please input focal length: ")
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
        ## Calculating warp matrix
        src_points = np.float32([[0,0], [self.new_width-1, 0], [0, self.new_height - 1]]) # top left, top right, bottom left
        dst_points = np.float32([[cx - self.new_width / 2.0, cy - self.new_height / 2.0], 
                                 [cx + self.new_width / 2.0, cy - self.new_height / 2.0],
                                 [cx - self.new_width / 2.0, cy + self.new_height / 2.0]])
            
        self.warp_mat = cv2.getAffineTransform(src_points, dst_points)
        

    def save_extrinsic_image(self):
        ## Saving AGI Format Poses and Images
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

        for ele in tqdm(self.xml_tree):
            name = ele.tag
            if name == "Photo":
                pose = load_xml_transform(ele.find('Pose'))
                image_path = os.path.basename(ele.find("ImagePath").text)

                if image_path not in img_basename_list:
                    continue
                    raise FileNotFoundError("Image %s not found" %image_path)
                
                if (pose == _default_matrix).all():
                    idx += 1
                    
                else:
                    r = Rotation.from_matrix(pose[:3, :3])
                    q = r.as_quat()
                    # print(idx)
                    self.agi_poses_quaternion_dict["poses"].append(np.array([pose[0, 3], pose[1, 3], pose[2, 3], q[0], q[1], q[2], q[3]]))
                    self.agi_poses_quaternion_dict["image_names"].append(image_path)

                    image_path = os.path.join(os.path.dirname(self.image_paths[0]), image_path)
                    #self.process_img(image_path)

                    idx += 1
        
        colmap_poses = convert_save_poses(self.agi_poses_quaternion_dict["poses"]) # in format of [qw, qx, qy, qz, tx, ty, tz]

        with open(self.extrinsic_path, "w+") as file:
            for idx,pose in enumerate(colmap_poses):
                image_name = self.agi_poses_quaternion_dict["image_names"][idx]
                output_line = f"{idx +1 } {pose[0]} {pose[1]} {pose[2]} {pose[3]} {pose[4]} {pose[5]} {pose[6]} 1 {image_name}\n No Content \n"
                file.write(output_line)      


    def save_pointcloud(self):
        downsample_scale = self.downsample_pcd
        if downsample_scale > 0 and downsample_scale < 1:
            print("Downsample pointcloud with scale %f" %downsample_scale)
            downsample_k = round(1 / downsample_scale)
            new_pcd = self.pcd.uniform_down_sample(every_k_points = downsample_k)
            o3d.io.write_point_cloud(self.pointcloud_path, new_pcd)
        else:
            print("Will not downsample pointcloud")
            o3d.io.write_point_cloud(self.pointcloud_path, self.pcd)
    
    def disposal_xml(self):
        for photogroup in self.photo_xml:
            photo = photogroup.find('Photo')
            img_path = photo.find("ImagePath").text
            folder_name = os.path.basename(os.path.dirname(img_path))

            if folder_name == "insta":
                self.xml_tree = photogroup


class agi2colmap_titan():
    def __init__(self, 
                 xml_path: str, titan_folder: str, ply_path: str, output_path: str, 
                 resize_img_scale : float = -1, 
                 resize_img_given_width : int = -1, 
                 recenter_img: bool = False, 
                 downsample_pcd: float = -1,
                 img_select_range: IndexRange = None):
        self.xml_path = xml_path
        self.titan_folder = titan_folder
        self.ply_path = ply_path
        self.output_path = output_path

        self.resize_img_scale = resize_img_scale
        self.resize_img_given_width = resize_img_given_width
        self.recenter_img = recenter_img
        self.downsample_pcd = downsample_pcd
        self.img_select_range = img_select_range

    def check_files(self):
        print("Checking essential files")
        try:
            xml_tree = ET.parse(self.xml_path)
            print("XML Ok.")
            self.photo_xml = xml_tree.getroot().find("Block").find("Photogroups")#.find("Photogroup")
        except:
            raise FileNotFoundError("XML doesn't exists")

        self.disposal_xml()

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

        # check images
        for index in range(0, 9):
            if index == 3:
                continue
            img_path = os.path.join(self.titan_folder, "titan_" + str(index))
            if not os.path.exists(img_path):
                raise FileNotFoundError("Image Folder Not Found")
        # self.images = []
        # self.image_names = []
        print("Checking images")
        if sys.platform == "win32":
            img_exts = ['png', 'jpg', 'jpeg']
        elif sys.platform == "linux":
            img_exts = ['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG']

        self.image_paths = []
        
        if self.img_select_range is None:
            for index in range(0, 9):
                for ext in img_exts:
                    if index == 3:
                        continue
                    images = natsorted(glob(os.path.join(self.titan_folder, "titan_" + str(index), "*." + ext)))
                    if len(images) != 0:
                        self.image_paths.append(images)
        elif self.img_select_range.start >= self.img_select_range.end:
            raise ValueError("img selection range error: starting index >= ending index")
        
        else:
            for index in range(0, 9):
                for ext in img_exts:
                    if index == 3:
                        continue
                    images = natsorted(glob(os.path.join(self.titan_folder, "titan_" + str(index), "*." + ext)))
                    if len(images) == 0:
                        continue
                    elif len(images) < self.img_select_range.end:
                        raise ValueError("img selection range error: ending index out of range")
                    else:
                        self.image_paths.append(images[self.img_select_range.start:(self.img_select_range.end + 1):self.img_select_range.step])
        ### self.image_paths: 8 * N_images
        if len(self.image_paths) == 0:
            raise FileNotFoundError("No Image Found")
        else:
            print("Images ok")
            self.raw_height, self.raw_width, _ = cv2.imread(self.image_paths[0][0]).shape

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

    def undistort_fisheye_single_cam(self, xml_path_list, new_width = 748, new_height = 748, fx = 884.59):
        undistorted_images = []
        if len(xml_path_list) != len(self.image_paths):
            raise ValueError("Xml numbers unequal to image folder numbers")
        for i in range(len(xml_path_list)):
            undistorting_cameras = undistort_cameras_opencv(image_path_list=self.image_paths[i], writing_mode=False)
            undistorting_cameras.read_xml(xml_path=xml_path_list[i])
            undistorted_images.append(undistorting_cameras.set_image_params_and_run(width=new_width, height=new_height, fx = fx))
        fx, fy = undistorting_cameras.get_focal()
        cx, cy = undistorting_cameras.get_cxcy()
        self.new_width, self.new_height = undistorting_cameras.get_shape()
        colmap_intrinsic_text = ["1", "PINHOLE", self.new_width, self.new_height, fx, fy, cx, cy]
        with open(self.intrinsic_path, "w+") as file:
            for i in range(len(colmap_intrinsic_text)):
                try:
                    file.write(colmap_intrinsic_text[i])
                except:
                    file.write(str(colmap_intrinsic_text[i]))
                file.write(" ")
            file.write("\n")
        self.colmap_intrinsic_text = colmap_intrinsic_text
        ### for test & debug
        # print(np.array(undistorted_images).shape)
        # print(np.array(self.image_paths).shape)
        # test_img_folder = os.path.join(self.output_path, "test_undistort_fishey_single_cam")
        # os.makedirs(test_img_folder, exist_ok=True)
        
        for i in range(len(self.image_paths)):
            for j in range(len(self.image_paths[i])):
                image_path = os.path.join(self.outimage_folder, os.path.basename(self.image_paths[i][j]))
                cv2.imwrite(image_path, undistorted_images[i][j])
    
    def undistort_fisheye_multi_cam(self, xml_path_list):
        undistorted_images = []
        colmap_intrinsic_text = []
        if len(xml_path_list) != len(self.image_paths):
            raise ValueError("Xml numbers unequal to image folder numbers")
        for i in range(len(xml_path_list)):
            undistorting_cameras = undistort_cameras_opencv(image_path_list=self.image_paths[i], writing_mode=False)
            undistorting_cameras.read_xml(xml_path=xml_path_list[i])
            undistorted_images.append(undistorting_cameras.set_image_params_and_run(fovX = 0.8))
            fx, fy = undistorting_cameras.get_focal()
            cx, cy = undistorting_cameras.get_cxcy()
            self.new_width, self.new_height = undistorting_cameras.get_shape()
            colmap_intrinsic_text.append([i, "PINHOLE", self.new_width, self.new_height, fx, fy, cx, cy])
        with open(self.intrinsic_path, "w+") as file:
            for i in range(len(colmap_intrinsic_text)):
                try:
                    file.write(colmap_intrinsic_text[i])
                except:
                    file.write(str(colmap_intrinsic_text[i]))
                file.write(" ")
                file.write("\n")

        self.colmap_intrinsic_text = colmap_intrinsic_text
        ### for tests and debug
        # print(np.array(undistorted_images).shape)
        # print(np.array(self.image_paths).shape)
        # test_img_folder = os.path.join(self.output_path, "test_undistort_fishey_multi_cam")
        # os.makedirs(test_img_folder, exist_ok=True)

        for i in range(len(self.image_paths)):
            for j in range(len(self.image_paths[i])):
                image_path = os.path.join(self.outimage_folder, os.path.basename(self.image_paths[i][j]))
                cv2.imwrite(image_path, undistorted_images[i][j])

    def save_extrinsic_image(self):
        ## Saving AGI Format Poses and Images
        self.agi_poses_quaternion_dict = {"poses":[], "image_names": [], "camera_id":[]}
        
        img_basename_list = []
        for i, img_subfolder in enumerate(self.image_paths):
            img_basename_list.append([])
            for img_path in img_subfolder:
                img_basename_list[i].append(os.path.basename(img_path)) # [8, N_images]
        _default_matrix = np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
        
        idx = 0

        # for ele in tqdm(self.photo_xml)
        for index, photo_xmls in enumerate(self.xml_tree_list):
            for ele in photo_xmls:
                name = ele.tag
                if name == "Photo":
                    pose = load_xml_transform(ele.find('Pose'))
                    image_path = os.path.basename(ele.find("ImagePath").text)
                    if image_path not in img_basename_list[index]:
                        continue
                        raise FileNotFoundError("Image %s not found" %image_path)
                    
                    if (pose == _default_matrix).all():
                        idx += 1
                        pass
                    else:
                        r = Rotation.from_matrix(pose[:3, :3])
                        q = r.as_quat()
                        # print(idx)
                        self.agi_poses_quaternion_dict["poses"].append(np.array([pose[0, 3], pose[1, 3], pose[2, 3], q[0], q[1], q[2], q[3]]))
                        self.agi_poses_quaternion_dict["image_names"].append(image_path)
                        self.agi_poses_quaternion_dict["camera_id"].append(index+1)

                        idx += 1
        colmap_poses = convert_save_poses(self.agi_poses_quaternion_dict["poses"]) # in format of [qw, qx, qy, qz, tx, ty, tz]
        if type(self.colmap_intrinsic_text[0]) != list:
            with open(self.extrinsic_path, "w+") as file:
                for idx,pose in enumerate(colmap_poses):
                    image_name = self.agi_poses_quaternion_dict["image_names"][idx]
                    output_line = f"{idx +1 } {pose[0]} {pose[1]} {pose[2]} {pose[3]} {pose[4]} {pose[5]} {pose[6]} 1 {image_name}\n No Content \n"
                    file.write(output_line)
        else:
            with open(self.extrinsic_path, "w+") as file:
                for idx,pose in enumerate(colmap_poses):
                    image_name = self.agi_poses_quaternion_dict["image_names"][idx]
                    camera_id = self.agi_poses_quaternion_dict["camera_id"][idx]
                    output_line = f"{idx +1 } {pose[0]} {pose[1]} {pose[2]} {pose[3]} {pose[4]} {pose[5]} {pose[6]} {camera_id} {image_name}\n No Content \n"
                    file.write(output_line)

    def save_pointcloud(self):
        downsample_scale = self.downsample_pcd
        if downsample_scale > 0 and downsample_scale < 1:
            print("Downsample pointcloud with scale %f" %downsample_scale)
            downsample_k = round(1 / downsample_scale)
            new_pcd = self.pcd.uniform_down_sample(every_k_points = downsample_k)
            o3d.io.write_point_cloud(self.pointcloud_path, new_pcd)
        else:
            print("Will not downsample pointcloud")
            o3d.io.write_point_cloud(self.pointcloud_path, self.pcd)

    def disposal_xml(self):
        xml_tree_list = ["0","1","2","4","5","6","7","8"]
        for photogroup in self.photo_xml:
            photo =  photogroup.find('Photo')
            img_path = photo.find("ImagePath").text
            folder_name = os.path.basename(os.path.dirname(img_path))

            if folder_name == "titan_0":
                xml_tree_list[0] = photogroup

            elif folder_name == "titan_1":
                xml_tree_list[1] = photogroup

            elif folder_name == "titan_2":
                xml_tree_list[2] = photogroup
            
            elif folder_name == "titan_4":
                xml_tree_list[3] = photogroup
            
            elif folder_name == "titan_5":
                xml_tree_list[4] = photogroup
            
            elif folder_name == "titan_6":
                xml_tree_list[5] = photogroup
            
            elif folder_name == "titan_7":
                xml_tree_list[6] = photogroup

            elif folder_name == "titan_8":
                xml_tree_list[7] = photogroup

        self.xml_tree_list = xml_tree_list
            