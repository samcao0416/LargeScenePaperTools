"""
This script is used for converting AGI calibration to colmap format,
and can be directly read by 3D gaussian splatting
Step 1. Check all the files may used
Step 2. Read XML File -> Get Image Path and Pose -> Saved in RAM
Step 3. Crop and resize Pointcloud
Step 4. Confirm Image Selection, Undistort and Resize Parameters
Step 5. Process Images
Step 6. Save Intrinsic and Extrinsic (images.txt, cameras.txt)
"""


import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
import os
import cv2
import open3d as o3d
import sys
from tqdm import tqdm
import warnings
from datetime import datetime
from natsort import natsorted
from glob import glob
from typing import NamedTuple
from utils.undistort_fisheye import undistort_single_cam_opencv
from utils.mesh_proj import MeshImageFilter
from utils.fisheye2multiples import Fisheye2Multiples
from joblib import Parallel, delayed

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


def convert_save_poses(pose: list):
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
    col_pose = np.array([qw, qx, qy, qz, tx, ty, tz])
    return col_pose


class Agi2Colmap():
    def __init__(self, 
                 xml_path: str, img_folder: str, ply_path: str, output_path: str, mesh_path: str = None, data_type: str = "insta"):
        self.xml_path = xml_path
        self.img_path = img_folder
        self.ply_path = ply_path
        self.output_path = output_path
        self.mesh_path = mesh_path

        if data_type == "insta":
            self.data_type = 1
        elif data_type == "titan":
            self.data_type = 0
        else:
            raise KeyError("Unrecognized data_type, should be \"insta\" or \"titan\"")
        
        # Initialize sub classes
        self.check_files = self.Check_Files(self)
        self.process_pcd = self.Process_Pcd(self)
        self.process_images = self.Process_Images(self)
        self.save_cameras = self.Save_Cameras(self)    

        # List all variables
        self.photo_xml = None
        self.xml_tree = None
        self.image_paths = None
        self.img_num_per_group = None
        self.pcd = None
        self.outimage_folder = None
        self.outother_folder = None
        self.instrinsic_path = None
        self.extrinsic_path  = None
        self.pointcloud_path = None
        self.mask_path = None
        
        self.expand_threshold = None
        self.bbox = None
        self.quad = None

        self.cam_pose_list = None

        self.raw_height = None
        self.raw_width = None
        self.new_height = None
        self.new_width = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

    class Check_Files():
        def __init__(self, outer_class):
            self.oc = outer_class

        def run(self):
            print("############################################################")
            print("[ INFO ] Checking essential files.")
            # Check XML
            self._check_xml()

            # Check Ply
            self._check_ply()

            # Check Images
            self._check_images()

            # Make Folders
            self._make_folders()
            print("############################################################")
        
        
        def _check_xml(self):
    
            try:
                xml_tree = ET.parse(self.oc.xml_path)
                print("[ INFO ] XML OK. ")
                self.oc.photo_xml = xml_tree.getroot().find("Block").find("Photogroups")
            except:
                raise FileNotFoundError("[ ERROR ] XML doesn't exists")
            
            self._disposal_xml()

        def _disposal_xml(self): 
            xml_tree_list = ["0","1","2","4","5","6","7","8"]
            for photogroup in self.oc.photo_xml:
                photo = photogroup.find('Photo')
                img_path = photo.find("ImagePath").text
                folder_name = os.path.basename(os.path.dirname(img_path))

                if self.oc.data_type:
                    if (folder_name == "insta") or (folder_name == "images"):
                        self.oc.xml_tree = photogroup
                else:
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
            
                    self.oc.xml_tree = xml_tree_list
        
        def _check_ply(self):
            try:
                pcd = o3d.io.read_point_cloud(self.oc.ply_path)
                if pcd is not None:
                    print("[ INFO ] Ply Ok.")
                else:
                        warnings.warn("Ply file doesn't exists, pointcloud process will be neglected, this might cause further issues.", RuntimeError)
                        pcd = None
            except:
                warnings.warn("[ WARN ] Ply file doesn't exists, pointcloud process will be neglected, this might cause further issues.", RuntimeError)
                pcd = None
            
            self.oc.pcd = pcd

        def _check_images(self):

            if self.oc.data_type:
                if not os.path.exists(self.oc.img_path):
                    raise FileNotFoundError("[ ERROR ] Image Folder Not Found")
            else:
                for index in range(0, 9):
                    if index == 3:
                        continue
                img_path = os.path.join(self.oc.img_path, "titan_" + str(index))
                if not os.path.exists(img_path):
                    raise FileNotFoundError("[ ERROR ] Image Folder Not Found")
            
            print("Checking Images.")

            if sys.platform == "win32":
                img_exts = ['png', 'jpg', 'jpeg']
            elif sys.platform == "linux":
                img_exts = ['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG']

            self.oc.image_paths = []
            
            if self.oc.data_type:
                for ext in img_exts:
                    self.oc.image_paths += natsorted(glob(os.path.join(self.oc.img_path, "*." + ext)))

                self.oc.img_num_per_group = len(self.oc.image_paths)
            else:
                for index in range(0, 9):
                    for ext in img_exts:
                        if index == 3:
                            continue
                        images = natsorted(glob(os.path.join(self.oc.img_path, "titan_" + str(index), "*." + ext)))
                        if len(images) != 0:
                            self.oc.image_paths.append(images)
                self.oc.img_num_per_group = len(self.oc.image_paths[0])
            if self.oc.img_num_per_group == 0:
                raise FileNotFoundError("[ ERROR ] No Image Found")
            else:
                
                print("[ INFO ] Images ok.")
                if self.oc.data_type:
                    self.oc.raw_height, self.oc.raw_width, _ = cv2.imread(self.oc.image_paths[0]).shape
                else:
                    self.oc.raw_height, self.oc.raw_width, _ = cv2.imread(self.oc.image_paths[0][0]).shape
        
        def _check_mesh(self):

            if self.oc.mesh_path is None:
                print("[ INFO ] Mesh Path is not given.")
            
            else:
                print("Checking Mesh.")
                if os.path.exists(self.oc.mesh_path):
                    print("Mesh ok.")

        def _make_folders(self):

            print("[ INFO ] Making Folders.")
            if self.oc.output_path == None:
                self.oc.output_path = "colmap_out_" + str("%02d" %datetime.now().month) + str("%02d" %datetime.now().day) + "_" + str("%02d" %datetime.hour) + str("%02d" %datetime.minute)
            os.makedirs(self.oc.output_path, exist_ok=True)

            self.oc.outimage_folder = os.path.join(self.oc.output_path, "images")
            self.oc.outother_folder = os.path.join(self.oc.output_path, "sparse", "0")

            os.makedirs(self.oc.outimage_folder, exist_ok=True)
            os.makedirs(self.oc.outother_folder, exist_ok=True)

            self.oc.intrinsic_path = os.path.join(self.oc.outother_folder, "cameras.txt")
            self.oc.extrinsic_path = os.path.join(self.oc.outother_folder, "images.txt")
            self.oc.pointcloud_path = os.path.join(self.oc.outother_folder, "points3D.ply")

    
    class Process_Pcd():
        def __init__(self, outer_class):
            self.oc = outer_class

        def run(self, downsample_scale: float = -1, bbox: list = None, quad: list = None, expand_threshold: float = 0.3):

            print("############################################################")
            print("[ INFO ] Processing Pointcloud.")

            if bbox is not None:
                self.crop_pcd(bbox=bbox, expand_threshold=expand_threshold)

            if quad is not None:
                self.crop_pcd_quad(quad = quad, expand_threshold=expand_threshold)

            self.downsample_pcd(downsample_scale=downsample_scale)

            self.save_pcd()
            print("############################################################")
        
        def downsample_pcd(self, downsample_scale : float = -1.0):
            if self.oc.pcd is None:
                print("[ WARN ] No PointCloud File Found.")
                return 0
            if downsample_scale > 0 and downsample_scale < 1:
                print("Downsample pointcloud with scale %f" %downsample_scale)
                downsample_k = round(1 / downsample_scale)
                self.oc.pcd = self.oc.pcd.uniform_down_sample(every_k_points = downsample_k)
                
            else:
                print("[ INFO ] Will not downsample pointcloud.")

        def crop_pcd(self, bbox: list = None, expand_threshold: float = 0.3):
            if self.oc.pcd is None:
                print("[ WARN ] No PointCloud File Found.")
                return 0
            if bbox is None:
                print("[ WARN ] No Bounding Box Given.")
                return 0
            else:
                min_bound = bbox[:3]
                max_bound = bbox[3:]

            # Check bounds validity
            if any(m >= M for m, M in zip(min_bound, max_bound)):
                print("Invalid clip box dimensions. Ensure min_bound < max_bound.")
                return 0
            
            # Check expand_threshold validity:
            if (type(expand_threshold) is not float) and (type(expand_threshold) is not int):
                print("Wrong Expand Type. Should be Float")
                print(expand_threshold, type(expand_threshold))
                return 0
            
            self.oc.expand_threshold = expand_threshold
            
            # update the bbox 
            bbox = np.array(bbox)
            bbox = bbox * (1 + expand_threshold)
            min_bound = bbox[:3]
            max_bound = bbox[3:]
            self.oc.bbox = bbox.tolist()

            # Crop the pointcloud with bbox
            print("[ INFO ] Crop Pointcloud with bbox:, ", self.oc.bbox)
            self.oc.pcd = self.oc.pcd.crop(
                o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound))
            
        
        def crop_pcd_quad(self, quad: list = None, expand_threshold: float = 0.0):
            # quad should be in the order of excatly clock-wise or counter-clock-wise
            # eg. [bottom_left, bottom_right, top_right, top_left] or [bottom_left, top_left, top_right, bottom_right]
            if self.oc.pcd is None:
                print("[ WARN ] No PointCloud File Found.")
                return 0
            if quad is None:
                print("[ WARN ] No Bounding Box Given.")
                return 0

            if (type(expand_threshold) is not float) and (type(expand_threshold) is not int):
                print("Wrong Expand Type. Should be Float")
                print(expand_threshold, type(expand_threshold))
                return 0 
            
            self.oc.expand_threshold = expand_threshold

            # update the quad
            self.oc.quad = self._expand_quad(quad, expand_threshold)

            print("[ INFO ] Crop Pointcloud with quad: ", self.oc.quad)

            self.oc.pcd = self._do_crop_with_quad(self.oc.pcd, self.oc.quad)

            

        def save_pcd(self):
            o3d.io.write_point_cloud(self.oc.pointcloud_path, self.oc.pcd)
            print("[ INFO ] Pointcloud Saved in %s" %self.oc.pointcloud_path)

        def _expand_quad(self, quad, expand_threshold):
            center_x = sum(q[0] for q in quad) / 4
            center_y = sum(q[1] for q in quad) / 4
            
            factor = 1 + expand_threshold
            expanded_quad = []
            for q in quad:
                new_x = center_x + factor * (q[0] - center_x)
                new_y = center_y + factor * (q[1] - center_y)
                expanded_quad.append((new_x, new_y))
            
            return expanded_quad

        def _do_crop_with_quad(self, pcd, quad):
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)

            new_points = []
            new_colors = []

            for index, point in enumerate(tqdm(points)):
                point_xy = point[0:2]
                if is_point_inside_quad(quad, point_xy):
                    new_points.append(point)
                    new_colors.append(colors[index])

            new_points = np.array(new_points)
            new_colors = np.array(new_colors)

            new_ply = o3d.geometry.PointCloud()
            new_ply.points = o3d.utility.Vector3dVector(new_points)
            new_ply.colors = o3d.utility.Vector3dVector(new_colors)

            return new_ply

    

    class Process_Images():
        def __init__(self, outer_class):
            self.oc = outer_class
            self.cal_percentage_with_mask = True
            self.D_mask = None # sam added for down view mask initiliazation

        def select_image(self, img_select_range: IndexRange = None):

            self.select_img_with_range(img_select_range=img_select_range)

            if self.oc.cam_pose_list is None:
                self._read_cams()

        def select_img_with_range(self, img_select_range: IndexRange = None):

            if img_select_range is not None:
                if img_select_range.start >= img_select_range.end:
                    raise ValueError("img selection range error: starting index >= ending index")
                
                if img_select_range.end >= self.oc.img_num_per_group:
                    raise ValueError("img selection range error: ending index out of range: ", self.oc.img_num_per_group)
                
                if self.oc.data_type:
                    image_paths = self.oc.image_paths[img_select_range.start:(img_select_range.end+1):img_select_range.step]
                    self.oc.image_paths = image_paths
                else:
                    image_paths = []
                    for index in range(0, 8):
                        image_paths.append(self.oc.image_paths[index][img_select_range.start:(img_select_range.end+1):img_select_range.step])

                    self.oc.image_paths = image_paths
               
            else:
                print("[ INFO ] No Image Selection Range Given. Will not select image by range.")

        def select_img_with_bbox(self, bbox: list = None):
            if self.oc.bbox is None and bbox is not None:
                self.oc.bbox = bbox

            elif self.oc.bbox is None and bbox is None:
                print("[ INFO ] No Bounding Box Given. Will not select image by bounding box.")
            
            else:
                cam_pose_list = []
                for cam_pose in self.oc.cam_pose_list:
                    if self._in_bbox(cam_pose):
                        cam_pose_list.append(cam_pose)

            self.oc.cam_pose_list = self.oc.cam_pose_list

        def select_img_with_mesh(self, bbox: list = None, quad:list = None, percentage_threshold: float = 0.8, save_mask = True):
            
            if self.oc.mesh_path is None:
                raise ValueError("[ ERROR] Mesh Path is None. Please Specify Mesh Path.")
            
            if self.oc.new_width is None or \
               self.oc.new_height is None or \
               self.oc.fx is None or \
               self.oc.fy is None or \
               self.oc.cx is None or \
               self.oc.cy is None:
                
                raise SystemError("Camera Parameters Not Set. Please Set Camera Parameters (set_cam_param) First.")
            
            if self.oc.bbox is None and bbox is not None:
                self.oc.bbox = bbox

            elif self.oc.bbox is None and bbox is None and self.oc.quad is None and quad is not None:
                self.oc.quad = quad

            if self.oc.bbox is None and bbox is None and self.oc.quad is None and quad is None:
                print("[ INFO ] No Bounding Box or Quad Given. Will not select image by bounding box.")

            else:

                print("############################################################")
                print("[ INFO ] Selecting Images with Mesh.")
                
                cam_pose_list = []
                
                mesh = MeshImageFilter(mesh_path=self.oc.mesh_path)

                print("[ INFO ] Read Mesh and Casting Rays.")
                mesh_raycast_scene = mesh.raycastscene
                
                cam_intrinsics = np.array([[self.oc.fx, 0, self.oc.cx],
                                           [0, self.oc.fy, self.oc.cy],
                                           [0,0,1]])
                
                mesh_process = mesh.meshprocess
                mesh_process.crop_mesh(bbox = self.oc.bbox, quad = self.oc.quad)

                if save_mask:
                    mask_folder = os.path.join(self.oc.output_path, "masks")
                    os.makedirs(mask_folder, exist_ok=True)

                    crop_mesh_path = os.path.join(self.oc.output_path, "cropped_mesh.obj")
                    mesh_process.save_cropped_mesh(path = crop_mesh_path)

                    crop_mesh = MeshImageFilter(mesh_path = crop_mesh_path, D_view_mask=self.D_mask)    # sam added for down view mask

                # debug_folder = os.path.join(self.oc.output_path, "debug") # for debug
                # os.makedirs(debug_folder, exist_ok=True) # for debug
                    
                print("[ INFO ] Selecting Images.")

                # mesh.renderimg.set_vis(cam_intrinsics, self.oc.new_width, self.oc.new_height) # for debug
                
                # for cam_pose in tqdm(self.oc.cam_pose_list): # TODO: add cache option, do not need to read mesh everytime
                def process_cam_pose(cam_pose):
                    cam_extrinsics = self._Quat2Matrix(cam_pose=cam_pose)
                    mesh_raycast_scene.CastScene(intrinsic=cam_intrinsics, 
                                                 extrinsic=cam_extrinsics,
                                                 width=self.oc.new_width,
                                                 height=self.oc.new_height)
                    mesh_raycast_scene.triangles_per_view()

                    # for debug
                    # mesh_raycast_scene.save_visible_triangles(path = os.path.join(debug_folder, os.path.basename(cam_pose.img_path)[:-4] + ".ply"))
                    
                    # mesh.renderimg.render_img(cam_extrinsics, path = os.path.join(debug_folder, os.path.basename(cam_pose.img_path))) # for debug

                    if self.cal_percentage_with_mask:
                        
                        crop_mesh.raycastscene.CastScene(intrinsic=cam_intrinsics,
                                                                extrinsic=cam_extrinsics,
                                                                width = self.oc.new_width,
                                                                height = self.oc.new_height)
                        mask_path = os.path.join(mask_folder, os.path.basename(cam_pose.img_path)[:-4] + ".png")

                        percentage = crop_mesh.raycastscene.cal_mask_percentage()

                        if percentage >= percentage_threshold:
                            cam_pose_list.append(cam_pose)
                            if save_mask:
                                crop_mesh.raycastscene.save_mask_depth(path = mask_path)


                    else:

                        percentage = mesh_process.calculate_percentage(indices_list=mesh_raycast_scene.filtered_indices)

                        if percentage >= percentage_threshold:
                            cam_pose_list.append(cam_pose)
                            if save_mask:
                                crop_mesh.raycastscene.CastScene(intrinsic=cam_intrinsics,
                                                                extrinsic=cam_extrinsics,
                                                                width = self.oc.new_width,
                                                                height = self.oc.new_height)
                                
                                mask_path = os.path.join(mask_folder, os.path.basename(cam_pose.img_path)[:-4] + ".png")

                                crop_mesh.raycastscene.save_mask_depth(path = mask_path)

                Parallel(n_jobs=1)(delayed(process_cam_pose)(cam_pose) for cam_pose in tqdm(self.oc.cam_pose_list))
                # mesh.renderimg.destroy_vis() # for debug

                

                self.oc.cam_pose_list = cam_pose_list

                    

        def set_cam_param(self, K_xml:list, new_width: int = 2000, new_height:int = 2000, fovX: float = 0.8, method: str = "undistort_fisheye", rad = np.pi / 4):

            if method == "undistort_fisheye":
            
                self.undistort_cams = undistort_single_cam_opencv(workspace = self.oc.outimage_folder, data_type= self.oc.data_type)
            
                self.undistort_cams.read_xml(K_xml_path=K_xml)

                self.undistort_cams.set_param(width=new_width, height=new_height, fovX=fovX)

                self.oc.new_width, self.oc.new_height = self.undistort_cams.get_shape()
                self.oc.fx, self.oc.fy = self.undistort_cams.get_focal()
                self.oc.cx, self.oc.cy = self.undistort_cams.get_cxcy()

            elif method == "fisheye2multiples":

                self.fish2multi_cams = Fisheye2Multiples(workspace = self.oc.outimage_folder, data_type= self.oc.data_type, rad = rad)
                self.fish2multi_cams.read_xml(K_xml_path=K_xml)
                self.fish2multi_cams.set_param(width=new_width, height=new_height, fovX=fovX)

                self.oc.new_width, self.oc.new_height = self.fish2multi_cams.get_shape()
                self.oc.fx, self.oc.fy = self.fish2multi_cams.get_focal()
                self.oc.cx, self.oc.cy = self.fish2multi_cams.get_cxcy()

            else:
                raise ValueError("Unrecognized Method. Should be \"undistort_fisheye\" or \"fisheye2multiples\"")

        def undistort_fisheye(self, write_mode = True):
            
            if self.oc.new_width is None or \
               self.oc.new_height is None or \
               self.oc.fx is None or \
               self.oc.fy is None or \
               self.oc.cx is None or \
               self.oc.cy is None:
                
                raise SystemError("Camera Parameters Not Set. Please Set Camera Parameters (set_cam_param) First.")

            if self.oc.cam_pose_list is None:
                raise ValueError("No Camera Pose List Found. i.e. self.cam_pose_list is None")
            
            print("[ INFO ] Undistorting Images. New Images will be saved in %s" %self.oc.outimage_folder)

            def process_cam_pose(cam_pose):
                self.undistort_cams.undistort_image(cam_pose=cam_pose, write_mode=write_mode)

            Parallel(n_jobs=1)(delayed(process_cam_pose)(cam_pose) for cam_pose in tqdm(self.oc.cam_pose_list))

        def fisheye2multiples(self, views:list = ["F", "L","R", "U", "D"], write_mode = True, fisheye_mask_path = None):
            # TODO: add a update mode, to judge whether to update the cam_pose_list
            if self.oc.new_width is None or \
               self.oc.new_height is None or \
               self.oc.fx is None or \
               self.oc.fy is None or \
               self.oc.cx is None or \
               self.oc.cy is None:
                
                raise SystemError("Camera Parameters Not Set. Please Set Camera Parameters (set_cam_param) First.")
            
            if self.oc.cam_pose_list is None:
                raise ValueError("No Camera Pose List Found. i.e. self.cam_pose_list is None")
            
            if fisheye_mask_path is None:
                fisheye_mask_path = "assets/titan_mask.png"
            
            cam_pose_new = []
            print(len(self.oc.cam_pose_list))

            ### Sam adding Down view mask
            if "D" in views:
                D_mask_cam_pose = self.CamPoseCol(titan_index=0,
                                                  img_path=fisheye_mask_path,
                                                  R = np.array([0,0,0,1]),
                                                  T = np.array([0,0,0]))
                self.D_mask = np.any(self.fish2multi_cams.fisheye_to_multiples(cam_pose = D_mask_cam_pose, views = ["D"], write_mode=False)[0] == 255, axis= -1)
            ### Sam ended

            # for cam_pose in tqdm(self.oc.cam_pose_list):
            #     self.fish2multi_cams.fisheye_to_multiples(cam_pose = cam_pose, views = views, write_mode=write_mode)
            #     cam_pose_new += self.fish2multi_cams.get_new_cam_pose_list()

            def process_cam_pose(cam_pose, cam_pose_new):
                self.fish2multi_cams.fisheye_to_multiples(cam_pose=cam_pose, views=views, write_mode=write_mode)
                cam_pose_new += self.fish2multi_cams.get_new_cam_pose_list()

            Parallel(n_jobs=1)(delayed(process_cam_pose)(cam_pose, cam_pose_new) for cam_pose in tqdm(self.oc.cam_pose_list))

            self.oc.cam_pose_list = cam_pose_new
            print(len(self.oc.cam_pose_list))
            

        def set_cal_percentage(self, cal_percentage_with_mask: bool = False):
            self.cal_percentage_with_mask = cal_percentage_with_mask


        def _read_cams(self):
            if self.oc.data_type:
                self._read_cams_insta()

            else:
                self._read_cams_titan()

        def _read_cams_insta(self):
            img_basename_list = []
            for img_path in self.oc.image_paths:
                img_basename_list.append(os.path.basename(img_path))

            insta_folder = os.path.dirname(self.oc.image_paths[0])

            self.oc.cam_pose_list = self._read_poses(photo_xmls = self.oc.xml_tree, img_basename_list=img_basename_list, image_folder=insta_folder, titan_index=None)

        def _read_cams_titan(self):
            
            img_basename_list = []
            
            for i, img_subfolder in enumerate(self.oc.image_paths):
            
                for img_path in img_subfolder:
                    img_basename_list.append(os.path.basename(img_path)) # [ N_images, ]

            cam_pose_list = []

            for index, photo_xmls in enumerate(self.oc.xml_tree):

                titan_folder = os.path.dirname(self.oc.image_paths[index][0])

                titan_index = int(titan_folder[-1])

                cam_pose_list += (self._read_poses(photo_xmls=photo_xmls,img_basename_list=img_basename_list, image_folder = titan_folder, titan_index=titan_index))

            self.oc.cam_pose_list = cam_pose_list
            
        def _read_poses(self, photo_xmls, img_basename_list, image_folder, titan_index):
            
            _null_matrix = np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])

            cam_pose_list = []

            if titan_index == "0" or 0:
                titan_index = 3
                
            for ele in photo_xmls:
                name = ele.tag
                if name == "Photo":
                    pose = load_xml_transform(ele.find('Pose'))
                    img_name = os.path.basename(ele.find("ImagePath").text)
                    
                    if img_name not in img_basename_list:
                        continue

                    if (pose == _null_matrix).all():
                        continue
                    
                    r = Rotation.from_matrix(pose[:3, :3])
                    q = r.as_quat()
                    q_col = np.array([q[0], q[1], q[2], q[3]])
                    t_col = np.array([pose[0, 3], pose[1, 3], pose[2, 3]])

                    image_path = os.path.join(image_folder, img_name)

                    cam_pose_col = self.CamPoseCol(titan_index=titan_index,
                                                   img_path=image_path,
                                                   R = q_col,
                                                   T = t_col)
                    cam_pose_list.append(cam_pose_col)

            return cam_pose_list

        def _in_bbox(self, cam_pose):
            T = cam_pose.T

            if T[0] < self.oc.bbox[0] or T[0] > self.oc.bbox[3]:
                return False
            elif T[1] < self.oc.bbox[1] or T[1] > self.oc.bbox[4]:
                return False
            elif T[2] < self.oc.bbox[2] or T[2] > self.oc.bbox[5]:
                return False
            else:
                return True 

        def _Quat2Matrix(self, cam_pose):
            
            ext_mat = np.eye(4)
            R = Rotation.as_matrix(Rotation.from_quat(cam_pose.R))
            ext_mat[:3, :3] = R
            ext_mat[:3, -1] = cam_pose.T
            return ext_mat

        class CamPoseCol(NamedTuple):
            titan_index: int
            img_path: str
            R: np.ndarray
            T: np.ndarray


    class Save_Cameras():
        def __init__(self, outer_class):
            self.oc = outer_class

        def run(self):
            
            self.save_intrinsic()
            self.save_extrinsic()

        def save_intrinsic(self):

            colmap_intrinsic_text = ["1", "PINHOLE", self.oc.new_width, self.oc.new_height, self.oc.fx, self.oc.fy, self.oc.cx, self.oc.cy]
            with open(self.oc.intrinsic_path, "w+") as file:
                for i in range(len(colmap_intrinsic_text)):
                    try:
                        file.write(colmap_intrinsic_text[i])
                    except:
                        file.write(str(colmap_intrinsic_text[i]))
                    file.write(" ")
                file.write("\n")
            print("[ INFO ] Intrinsic Saved in %s" %self.oc.intrinsic_path)

        def save_extrinsic(self):

            with open(self.oc.extrinsic_path, "w+") as file:
                for idx, cam_pose in enumerate(self.oc.cam_pose_list):
                    image_name = os.path.basename(cam_pose.img_path)
                    pose = convert_save_poses(cam_pose.T.tolist() + cam_pose.R.tolist())
                    output_line = f"{idx +1 } {pose[0]} {pose[1]} {pose[2]} {pose[3]} {pose[4]} {pose[5]} {pose[6]} 1 {image_name}\n No Content \n"
                    file.write(output_line)      
            
            print("[ INFO ] Extrinsic Saved in %s" %self.oc.extrinsic_path)







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






   



    
    
    

    
    
    

    
    