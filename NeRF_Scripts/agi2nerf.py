import xml.etree.ElementTree as ET
import math
import cv2
import numpy as np
import os
import json
import warnings
from tqdm import tqdm

from datetime import datetime
from natsort import natsorted
from glob import glob
from typing import NamedTuple
from utils.undistort_fisheye import undistort_cameras_opencv
import sys

###############################################################################
# START
# code taken from https://github.com/NVlabs/instant-ngp
#Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c)**2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa+ta*da+ob+tb*db) * 0.5, denom

def central_point(out):
    # find a central point they are all looking at
    print("computing center of attention...")
    totw = 0.0
    totp = np.array([0.0, 0.0, 0.0])
    for f in out["frames"]:
        mf = np.array(f["transform_matrix"])[0:3,:]
        for g in out["frames"]:
            mg = g["transform_matrix"][0:3,:]
            p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
            if w > 0.0001:
                totp += p*w
                totw += w
    totp /= totw
    print(totp) # the cameras are looking at totp
    for f in out["frames"]:
        f["transform_matrix"][0:3,3] -= totp
        f["transform_matrix"] = f["transform_matrix"].tolist()
    return out

def sharpness(imagePath):
    image = cv2.imread(imagePath)
    if image is None:
        print("Image not found:", imagePath)
        return 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm

#END
###############################################################################

#Copyright (C) 2022, Enrico Philip Ahlers. All rights reserved.

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

def reflectZ():
    return [[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]]

def reflectY():
    return [[1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]

def matrixMultiply(mat1, mat2):
    return np.array([[sum(a*b for a,b in zip(row, col)) for col in zip(*mat2)] for row in mat1])

class agi2nerf_insta():
    def __init__(self, 
                 xml_path:str, img_folder: str, output_path: str,
                 resize_img_scale: float = -1,
                 resize_img_given_width: int = -1, 
                 recenter_img: bool = False,
                 img_select_range: list = None):
        self.xml_path = xml_path
        self.img_path = img_folder
        self.output_path = output_path

        self.resize_img_scale = resize_img_scale
        self.resize_img_given_width = resize_img_given_width
        self.recenter_img = recenter_img
        self.img_select_range = img_select_range

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

        # check images
        if not os.path.exists(self.img_path):
            raise FileNotFoundError("Image Folder Not Found")
        
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
                self.image_paths += natsorted(glob(os.path.join(self.img_path, "*." + ext)))[self.img_select_range.start:(self.img_select_range.end + 1)]

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
        os.makedirs(self.outimage_folder, exist_ok=True)
        
        self.transform_path = os.path.join(self.output_path, "transforms.json")

    def undistort_fisheye(self, xml_path, new_width = 1500, new_height = 1500, fovX = 0.8):
        undistorting_cameras = undistort_cameras_opencv(image_path_list=self.image_paths, writing_mode=False)
        undistorting_cameras.read_xml(xml_path=xml_path)
        undistorted_images = undistorting_cameras.set_image_params_and_run(fovX = fovX)
        fx, fy = undistorting_cameras.get_focal()
        cx, cy = undistorting_cameras.get_cxcy()
        self.new_width, self.new_height = undistorting_cameras.get_shape()

        self.intrinsic_list = [fx, fy, cx, cy]

        for i in range(len(self.image_paths)):
            image_path = os.path.join(self.outimage_folder, os.path.basename(self.image_paths[i]))
            cv2.imwrite(image_path, undistorted_images[i])

    def save_extrinsic_json(self):
        ## Saving Poses and Images
        self.agi_poses_nerf_dict = {"poses":[], "image_names":[]}

        img_basename_list = []
        for img_path in self.image_paths:
            img_basename_list.a(os.path.basename(img_path))
        
        _default_matrix = np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
        
        idx = 0

        for ele in tqdm(self.xml_tree):
            name = ele.tag
            if name == "Photo":
                pose = load_xml_transform(ele.find('Pose'))
                image_path = os.path.basename(ele.find("ImagePath").text)
                if image_path not in img_basename_list:
                    continue

                if (pose == _default_matrix).all():
                    idx += 1
                
                else:
                    pose = pose[[2, 0, 1, 3], :]
                    pose_nerf = matrixMultiply(matrixMultiply(pose, reflectZ()), reflectY())

                    self.agi_poses_nerf_dict["poses"].append(pose_nerf.tolist())
                    self.agi_poses_nerf_dict["image_names"].append(image_path)

                    idx += 1

        transforms = {
                'w': self.new_width,
                'h': self.new_height,
                'aabb_scale': 16,
                "fl_x": self.intrinsic_list[0],
                "fl_y": self.intrinsic_list[1],
                "cx": self.intrinsic_list[2],
                "cy": self.intrinsic_list[3],
                "camera_angle_x": math.atan(float(self.new_width) / (float(self.intrinsic_list[0]) * 2)) * 2,
                "camera_angle_y": math.atan(float(self.new_height) / (float(self.intrinsic_list[1]) * 2)) * 2,
                "frames":[]
                
            }
        for idx, pose in enumerate(self.agi_poses_nerf_dict["poses"]):
            image_path = "images/" + self.agi_poses_nerf_dict["image_names"][idx]
            transforms["frames"].append({
                "file_path": image_path,
                "transform_matrix": pose.tolist()
            })

    def disposal_xml(self):
        for photogroup in self.photo_xml:
            photo = photogroup.find('Photo')
            img_path = photo.find("ImagePath").text
            folder_name = os.path.basename(os.path.dirname(img_path))

            if folder_name == "insta":
                self.xml_tree = photogroup


class agi2nerf_titan():
    def __init__(self,
                 xml_path:str, titan_folder: str, output_path: str,
                 resize_img_scale: float = -1,
                 resize_img_given_width : int = -1, 
                 recenter_img: bool = False, 
                 img_select_range: list = None):
        self.xml_path = xml_path
        self.titan_folder = titan_folder
        self.output_path = output_path

        self.resize_img_scale = resize_img_scale
        self.resize_img_given_width = resize_img_given_width
        self.recenter_img = recenter_img
        self.img_select_range = img_select_range

    def check_files(self):

        # check xml
        print("Checking essential files")
        try:
            xml_tree = ET.parse(self.xml_path)
            print("XML Ok.")
            self.photo_xml = xml_tree.getroot().find("Block").find("Photogroups")
        except:
            raise FileNotFoundError("XML doesn't exists")
        
        self.disposal_xml()

        # check images
        for index in range(0, 9):
            if index == 3:
                continue
            img_path = os.path.join(self.titan_folder, "titan_" + str(index))
            if not os.path.exists(img_path):
                raise FileNotFoundError("Image Folder Not Found")
            
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
            self.output_path = "nerf_out_" + str("%02d" %datetime.now().month) + str("%02d" %datetime.now().day) + "_" + str("%02d" %datetime.hour) + str("%02d" %datetime.minute)
        os.makedirs(self.output_path, exist_ok=True)

        self.outimage_folder = os.path.join(self.output_path, "images")
        os.makedirs(self.outimage_folder, exist_ok=True)
        
        self.transform_path = os.path.join(self.output_path, "transforms.json")

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

        self.intrinsic_list = [fx, fy, cx, cy]

        for i in range(len(self.image_paths)):
            for j in range(len(self.image_paths[i])):
                image_path = os.path.join(self.outimage_folder, os.path.basename(self.image_paths[i][j]))
                cv2.imwrite(image_path, undistorted_images[i][j])

    def undistort_fisheye_multi_cam(self, xml_path_list):
        undistorted_images = []
        self.intrinsic_list = []
        if len(xml_path_list) != len(self.image_paths):
            raise ValueError("Xml numbers unequal to image folder numbers")
        for i in range(len(xml_path_list)):
            undistorting_cameras = undistort_cameras_opencv(image_path_list=self.image_paths[i], writing_mode=False)
            undistorting_cameras.read_xml(xml_path=xml_path_list[i])
            undistorted_images.append(undistorting_cameras.set_image_params_and_run(fovX = 0.8))
            fx, fy = undistorting_cameras.get_focal()
            cx, cy = undistorting_cameras.get_cxcy()
            self.new_width, self.new_height = undistorting_cameras.get_shape()
            self.insintric_list.append([fx, fy, cx, cy])

        for i in range(len(self.image_paths)):
            for j in range(len(self.image_paths[i])):
                image_path = os.path.join(self.outimage_folder, os.path.basename(self.image_paths[i][j]))
                cv2.imwrite(image_path, undistorted_images[i][j])
    
    def save_extrinsic_json(self):
        ## Saving Poses and Images
        self.agi_poses_nerf_dict = {"poses":[], "image_names":[], "camera_id":[]}

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

        for index, photo_xmls in enumerate(self.xml_tree_list):
            for ele in photo_xmls:
                name = ele.tag
                if name == "Photo":
                    pose = load_xml_transform(ele.find('Pose'))
                    image_path = os.path.basename(ele.find("ImagePath").text)
                    if image_path not in img_basename_list[index]:
                        continue

                    if (pose == _default_matrix).all():
                        idx += 1
                        pass
                    else:
                        pose = pose[[2, 0, 1, 3], :]
                        pose_nerf = matrixMultiply(matrixMultiply(pose, reflectZ()), reflectY())

                        self.agi_poses_nerf_dict["poses"].append(pose_nerf)
                        self.agi_poses_nerf_dict["image_names"].append(image_path)
                        self.agi_poses_nerf_dict["camera_id"].append(index+1)

                        idx += 1
        
        if type(self.intrinsic_list[0]) != list :
            transforms = {
                'w': self.new_width,
                'h': self.new_height,
                'aabb_scale': 16,
                "fl_x": self.intrinsic_list[0],
                "fl_y": self.intrinsic_list[1],
                "cx": self.intrinsic_list[2],
                "cy": self.intrinsic_list[3],
                "camera_angle_x": math.atan(float(self.new_width) / (float(self.intrinsic_list[0]) * 2)) * 2,
                "camera_angle_y": math.atan(float(self.new_height) / (float(self.intrinsic_list[1]) * 2)) * 2,
                "frames":[]
                
            }
            for idx, pose in enumerate(self.agi_poses_nerf_dict["poses"]):
                image_path = "images/" + self.agi_poses_nerf_dict["image_names"][idx]
                transforms["frames"].append({
                    "file_path": image_path,
                    "transform_matrix": pose.tolist()
                })
        
        else:
            transforms = {
                'w': self.new_width,
                'h': self.new_height,
                "aabb_scale": 16,
                "frames":[]
            }
            for idx, pose in enumerate(self.agi_poses_nerf_dict["poses"]):
                image_path = "images/" + self.agi_poses_nerf_dict["image_names"][idx]
                camera_id = self.agi_poses_nerf_dict["camera_id"][idx]
                fl_x = self.intrinsic_list[camera_id][0]
                fl_y = self.intrinsic_list[camera_id][1]
                cx = self.intrinsic_list[camera_id][2]
                cy = self.intrinsic_list[camera_id][3]
                K = [[fl_x, 0, cx],[0, fl_y, cy],[0, 0, 1]]
                transforms["frames"].append({
                    "K": K,
                    "fl_x": self.intrinsic_list[camera_id][0],
                    "fl_y": self.intrinsic_list[camera_id][1],
                    "cx": self.intrinsic_list[camera_id][2],
                    "cy": self.intrinsic_list[camera_id][3],
                    "file_path": image_path,
                    "transform_matrix": pose.tolist()
                })
        with open(self.transform_path, 'w') as f:
            json.dump(transforms, f, indent=4)


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