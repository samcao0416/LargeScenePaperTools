"""
Step 1. Check all the files may used
Step 2. Manipulate images (Undistort_fisheye or fisheye2multiples)
Step 3. Raycast and get corresponding relations matrix and save it
    matrix_1:
        row_num    : view_num
        column_num : triangle_num
        element    : 1 (seen) / 0 (Unseen)

Step 4. Read Quad List
Step 5. Make Folders
Step 6. Crop Pcds and save
Step 7. Crop Mesh and get corresponding matrix

    matrix_2:
        row_num    : triangle_num
        column_num : block_num
        element    : 1 (in) / 0 (not in)
    matrix_3:
        row_num    : view_num
        colnum_num : block_num
        element    : num of triangles in the block are seen

    matrix_3 = matrix_1 * matrix_2

    save matrix 2 & 3

    matrix_4:
        row_num    : view_num
        colnum_num : block_num
        element    : ratio of triangles in the block are seen

        matrix_4 = matrix_3 ./ matrix_2.sum(axis=0).repeat(view_num).reshape(block_num, -1).T 分母是每个block的三角形数

    matrix_5:
        row_num    : view_num
        colnum_num : block_num
        element    : 1(kept) / 0 (not kept)

Step 8. Use the matrix to select images
Step 9. Save Images
Step 10. Save Cameras
"""
"""
cache_folder:
    - cache
        - images
        - matrixes
        - quads
        cam_pose_list.pkl
"""

import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
import os
import cv2
import open3d as o3d
import sys
import pickle
import shutil
from tqdm import tqdm
import warnings
from datetime import datetime
from natsort import natsorted
from glob import glob
from typing import NamedTuple
from utils.undistort_fisheye import undistort_single_cam_opencv
from utils.mesh_proj import MeshImageFilter
from utils.fisheye2multiples2 import Fisheye2Multiples
from joblib import Parallel, delayed

class IndexRange(NamedTuple):
    start : int
    end   : int
    step  : int = 1

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
                 xml_path: str = None, 
                 img_folder: str = None, 
                 ply_path: str = None, 
                 mesh_path: str = None,
                 point_pick_txt_path: str = None,
                 data_type: str = "insta",
                 output_path: str = None,
                 ):
        self.xml_path = xml_path
        self.img_path = img_folder
        self.ply_path = ply_path
        self.mesh_path = mesh_path
        self.point_pick_txt_path = point_pick_txt_path
        self.output_path = output_path

        if data_type == "insta":
            self.data_type = 1
        elif data_type == "titan":
            self.data_type = 0
        else:
            raise KeyError("data_type should be \"insta\" or \"titan\"")
        
        if self.xml_path is None and \
            self.img_path is None and \
            self.ply_path is None and \
            self.mesh_path is None:
            raise ValueError("Not Enough Input")
        
        # Initialize sub classes
        self.check_files = self.Check_Files(self)
        self.process_images = self.Process_Images(self)
        self.process_blocks = self.Process_Blocks(self)

    class Check_Files():
        def __init__(self, parent_class):
            self.st = parent_class

        def run(self):
            print("############################################################")
            print("[ INFO ] Checking essential files.")
            # Check XML
            self._check_xml()

            # Check Ply
            self._check_ply()

            # Check Images
            self._check_images()

            # Check Mesh
            self._check_mesh()

            # Check Txt 
            self._check_text()

            # Make Cache Folder
            self._make_cache_folders()
            print("############################################################")

        def load_cache(self, check_images = False):
            print("############################################################")
            print("[ INFO ] Checking and loading cache files.")

            self._load_cam_pose_list()

            self._check_ply()

            self._check_mesh()

            if check_images:
                self._check_each_img()

        def _check_xml(self):
    
            try:
                xml_tree = ET.parse(self.st.xml_path)
                print("[ INFO ] XML OK. ")
                self.st.photo_xml = xml_tree.getroot().find("Block").find("Photogroups")
            except:
                raise FileNotFoundError("[ ERROR ] XML doesn't exists")
            
            self._disposal_xml()

        def _disposal_xml(self): 
            xml_tree_list = ["0","1","2","4","5","6","7","8"]
            for photogroup in self.st.photo_xml:
                photo = photogroup.find('Photo')
                img_path = photo.find("ImagePath").text
                folder_name = os.path.basename(os.path.dirname(img_path))

                if self.st.data_type:
                    if (folder_name == "insta") or (folder_name == "images"):
                        self.st.xml_tree = photogroup
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
            
                    self.st.xml_tree = xml_tree_list
        
        def _check_ply(self):
            try:
                pcd = o3d.io.read_point_cloud(self.st.ply_path)
                if pcd is not None:
                    print("[ INFO ] Ply Ok.")
                else:
                        warnings.warn("Ply file doesn't exists, pointcloud process will be neglected, this might cause further issues.", RuntimeError)
                        pcd = None
            except:
                warnings.warn("[ WARN ] Ply file doesn't exists, pointcloud process will be neglected, this might cause further issues.", RuntimeError)
                pcd = None
            
            self.st.pcd = pcd

        def _check_images(self):

            if self.st.data_type:
                if not os.path.exists(self.st.img_path):
                    raise FileNotFoundError("[ ERROR ] Image Folder Not Found")
            else:
                for index in range(0, 9):
                    if index == 3:
                        continue
                img_path = os.path.join(self.st.img_path, "titan_" + str(index))
                if not os.path.exists(img_path):
                    raise FileNotFoundError("[ ERROR ] Image Folder Not Found")
            
            print("[ INFO ]Checking Images.")

            if sys.platform == "win32":
                img_exts = ['png', 'jpg', 'jpeg']
            elif sys.platform == "linux":
                img_exts = ['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG']

            self.st.image_paths = []
            
            if self.st.data_type:
                for ext in img_exts:
                    self.st.image_paths += natsorted(glob(os.path.join(self.st.img_path, "*." + ext)))

                self.st.img_num_per_group = len(self.st.image_paths)
            else:
                for index in range(0, 9):
                    for ext in img_exts:
                        if index == 3:
                            continue
                        images = natsorted(glob(os.path.join(self.st.img_path, "titan_" + str(index), "*." + ext)))
                        if len(images) != 0:
                            self.st.image_paths.append(images)
                self.st.img_num_per_group = len(self.st.image_paths[0])
            if self.st.img_num_per_group == 0:
                raise FileNotFoundError("[ ERROR ] No Image Found")
            else:
                
                print("[ INFO ] Images ok.")
                if self.st.data_type:
                    self.st.raw_height, self.st.raw_width, _ = cv2.imread(self.st.image_paths[0]).shape
                else:
                    self.st.raw_height, self.st.raw_width, _ = cv2.imread(self.st.image_paths[0][0]).shape
        
        def _check_text(self):
            
            if self.st.point_pick_txt_path is None:
                print("[ WARN ] No point picking list is given.")

            else:
                if os.path.exists(self.st.point_pick_txt_path):
                    print("[ INFO ] TXT OK.")
                else:
                    raise FileExistsError("Point Picking Txt File does not exists.")

        def _check_mesh(self):

            if self.st.mesh_path is None:
                print("[ INFO ] Mesh Path is not given.")
            
            else:
                print("Checking Mesh.")
                if os.path.exists(self.st.mesh_path):
                    print("Mesh ok.")

        def _make_cache_folders(self):

            print("[ INFO ] Making Cache Folders.")
            if self.st.output_path == None:
                self.st.output_path = "colmap_out_" + str("%02d" %datetime.now().month) + str("%02d" %datetime.now().day) + "_" + str("%02d" %datetime.hour) + str("%02d" %datetime.minute)

            self.st.cache_path = os.path.join(self.st.output_path, "cache")

            self.st.cache_img_path = os.path.join(self.st.cache_path, "images")
            self.st.cache_matrix_path = os.path.join(self.st.cache_path, "matrices")
            self.st.cache_quad_path = os.path.join(self.st.cache_path, "quads")

            os.makedirs(self.st.cache_img_path, exist_ok=True)
            os.makedirs(self.st.cache_matrix_path, exist_ok=True)
            os.makedirs(self.st.cache_quad_path, exist_ok=True)

        def _load_cam_pose_list(self):

            campose_list_path = os.path.join(self.st.cache_path, "cam_pose_list.pkl")

            with open(campose_list_path, 'rb') as f:
                self.st.cam_pose_list = pickle.load(f)

        def _check_each_img(self):
            print("[ INFO ] Checking Each Image.")
            new_cam_post_list = []
            for cam_pose in tqdm(self.st.cam_pose_list):
                img_path = os.path.join(self.st.cache_path, "images", os.basename(cam_pose.img_path))
                try:
                    tmp_img = cv2.imread(img_path)
                    if tmp_img is not None:
                        new_cam_post_list.append(cam_pose)
                    else:
                        print("[ WARN ] Image %s doesn't exists, will be neglected." %img_path)
                except:
                    print("[ WARN ] Image %s doesn't exists, will be neglected." %img_path)
            self.st.cam_pose_list = new_cam_post_list
            print("[ INFO ] Image Check Done.")

    class Process_Images():
        def __init__(self, parent_class):
            self.st = parent_class
            self.cal_percentage_with_mask = True # TODO: False?
            self.D_mask = None
        
        def read_cams(self):
            if self.st.data_type:
                self._read_cams_insta()

            else:
                self._read_cams_titan()

        def select_image(self, img_select_range: IndexRange = None):

            self.select_img_with_range(img_select_range = img_select_range)

        def select_img_with_range(self, img_select_range: IndexRange = None):

            if img_select_range is not None:
                if img_select_range.start >= img_select_range.end:
                    raise ValueError("img selection range error: starting index >= ending index")
                
                if img_select_range.end >= self.st.img_num_per_group:
                    raise ValueError("img selection range error: ending index out of range: ", self.st.img_num_per_group)
                
                if self.st.data_type:
                    # image_paths = self.st.image_paths[img_select_range.start:(img_select_range.end+1):img_select_range.step]
                    # self.st.image_paths = image_paths
                    self.st.cam_pose_list = self.st.cam_pose_list[img_select_range.start:(img_select_range.end+1):img_select_range.step]
                else:
                    image_paths = []
                    # TODO: Finishe titan case
                    for index in range(0, 8):
                        image_paths.append(self.st.image_paths[index][img_select_range.start:(img_select_range.end+1):img_select_range.step])

                    self.st.image_paths = image_paths
               
            else:
                print("[ INFO ] No Image Selection Range Given. Will not select image by range.")

        def set_cam_param(self, K_xml:list, new_width: int = 2000, new_height:int = 2000, fovX: float = 0.8, method: str = "undistort_fisheye", \
                          rad = np.pi / 4, fisheye_mask_path:str = "assets/titan_mask.png", views:list = ["F", "L","R", "U", "D"]): # second lines are parameter for method "fisheye2muliples"
            
            if method == "undistort_fisheye":
                self.undistort_cams = undistort_single_cam_opencv(workspace = self.st.outimage_folder, data_type= self.st.data_type)
            
                self.undistort_cams.read_xml(K_xml_path=K_xml)

                self.undistort_cams.set_param(width=new_width, height=new_height, fovX=fovX)

                self.st.new_width, self.st.new_height = self.undistort_cams.get_shape()
                self.st.fx, self.st.fy = self.undistort_cams.get_focal()
                self.st.cx, self.st.cy = self.undistort_cams.get_cxcy()

            elif method == "fisheye2multiples":
                self.fish2multi_cams = Fisheye2Multiples(workspace = self.st.cache_img_path, data_type= self.st.data_type, rad = rad)
                self.fish2multi_cams.read_xml(K_xml_path=K_xml)
                self.fish2multi_cams.set_param(width=new_width, height=new_height, fovX=fovX)
                self.fish2multi_cams.generate_points(views = views)

                self.st.new_width, self.st.new_height = self.fish2multi_cams.get_shape()
                self.st.fx, self.st.fy = self.fish2multi_cams.get_focal()
                self.st.cx, self.st.cy = self.fish2multi_cams.get_cxcy()

                # D_mask_cam_pose = self.CamPoseCol(titan_index=0,
                #                                   img_path=fisheye_mask_path,
                #                                   R = np.array([0,0,0,1]),
                #                                   T = np.array([0,0,0]))
                # self.D_mask = np.any(self.fish2multi_cams.fisheye_to_multiples(cam_pose = D_mask_cam_pose, views = ["D"], write_mode=False)[0] == 255, axis= -1) TODO: Bug here

            else:
                raise ValueError("Unrecognized Method. Should be \"undistort_fisheye\" or \"fisheye2multiples\"")
            
        def undistort_fisheye(self, write_mode = True):
            
            if self.st.new_width is None or \
               self.st.new_height is None or \
               self.st.fx is None or \
               self.st.fy is None or \
               self.st.cx is None or \
               self.st.cy is None:
                
                raise SystemError("Camera Parameters Not Set. Please Set Camera Parameters (set_cam_param) First.")

            if self.st.cam_pose_list is None:
                raise ValueError("No Camera Pose List Found. i.e. self.cam_pose_list is None")
            
            print("[ INFO ] Undistorting Images. New Images will be saved in %s" %self.st.outimage_folder)

            def process_cam_pose(cam_pose):
                self.undistort_cams.undistort_image(cam_pose=cam_pose, write_mode=write_mode)

            Parallel(n_jobs=1)(delayed(process_cam_pose)(cam_pose) for cam_pose in tqdm(self.st.cam_pose_list))

        def fisheye2multiples(self, views:list = ["F", "L","R", "U", "D"], write_mode = True, fisheye_mask_path = None):
            # TODO: add a update mode, to judge whether to update the cam_pose_list
            if self.st.new_width is None or \
               self.st.new_height is None or \
               self.st.fx is None or \
               self.st.fy is None or \
               self.st.cx is None or \
               self.st.cy is None:
                
                raise SystemError("Camera Parameters Not Set. Please Set Camera Parameters (set_cam_param) First.")
            
            if self.st.cam_pose_list is None:
                raise ValueError("No Camera Pose List Found. i.e. self.cam_pose_list is None")
            
            if fisheye_mask_path is None:
                fisheye_mask_path = "assets/titan_mask.png"
            
            cam_pose_new = []
            print(len(self.st.cam_pose_list))

            # for cam_pose in tqdm(self.st.cam_pose_list):
            #     self.fish2multi_cams.fisheye_to_multiples(cam_pose = cam_pose, views = views, write_mode=write_mode)
            #     cam_pose_new += self.fish2multi_cams.get_new_cam_pose_list()

            def process_cam_pose(cam_pose, cam_pose_new):
                self.fish2multi_cams.fisheye_to_multiples(cam_pose=cam_pose, write_mode=write_mode)
                cam_pose_new += self.fish2multi_cams.get_new_cam_pose_list()

            Parallel(n_jobs=1)(delayed(process_cam_pose)(cam_pose, cam_pose_new) for cam_pose in tqdm(self.st.cam_pose_list))

            self.st.cam_pose_list = cam_pose_new
            print(len(self.st.cam_pose_list))

        def _read_cams_insta(self):
            img_basename_list = []
            for img_path in self.st.image_paths:
                img_basename_list.append(os.path.basename(img_path))

            insta_folder = os.path.dirname(self.st.image_paths[0])

            self.st.cam_pose_list = self._read_poses(photo_xmls = self.st.xml_tree, img_basename_list=img_basename_list, image_folder=insta_folder, titan_index=None)

        def _read_cams_titan(self):
            
            img_basename_list = []
            
            for i, img_subfolder in enumerate(self.st.image_paths):
            
                for img_path in img_subfolder:
                    img_basename_list.append(os.path.basename(img_path)) # [ N_images, ]

            cam_pose_list = []

            for index, photo_xmls in enumerate(self.st.xml_tree):

                titan_folder = os.path.dirname(self.st.image_paths[index][0])

                titan_index = int(titan_folder[-1])

                cam_pose_list += (self._read_poses(photo_xmls=photo_xmls,img_basename_list=img_basename_list, image_folder = titan_folder, titan_index=titan_index))

            self.st.cam_pose_list = cam_pose_list
            
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
        
        class CamPoseCol(NamedTuple):
            titan_index: int
            img_path: str
            R: np.ndarray
            T: np.ndarray
        
    class Process_Blocks():

        def __init__(self, parent_class):

            self.st = parent_class

        def run_default(self, quad_expand_threshold=0.3, img_selection_threshold=0.005):
            self.get_view_triangle_matrix()
            self.get_blocks(expand_threshold=quad_expand_threshold)
            self.make_folders()
            self.crop_pcds()
            self.crop_mesh_get_matrix2()
            self.select_images(threshold=img_selection_threshold)

        def get_view_triangle_matrix(self):
            print("[ INFO ] Getting View Triangle Matrix.")
            view_num = len(self.st.cam_pose_list)

            mesh = MeshImageFilter(mesh_path=self.st.mesh_path)

            self.st.mesh = mesh
            
            self.st.triangle_num = np.asarray((mesh.mesh.triangles)).shape[0]

            #matrix_1 = np.zeros((view_num, triangle_num), dtype=np.uint8)

            mesh_raycast_scene = mesh.raycastscene

            cam_intrinsics = np.array([[self.st.fx, 0, self.st.cx],
                                           [0, self.st.fy, self.st.cy],
                                           [0,0,1]])
            
            for index, cam_pose in enumerate(tqdm(self.st.cam_pose_list)):
                
                cam_extrinsics = self._Quat2Matrix(cam_pose=cam_pose)
                mesh_raycast_scene.CastScene(intrinsic=cam_intrinsics, 
                                                 extrinsic=cam_extrinsics,
                                                 width=self.st.new_width,
                                                 height=self.st.new_height)
                hit_triangles_index = mesh_raycast_scene.get_hit_triangles_index()
                
                matrix_1_path = os.path.join(self.st.cache_matrix_path, "matrix_1_%03d.npy" %(index))
                np.save(matrix_1_path, hit_triangles_index)
            
            

        def _Quat2Matrix(self, cam_pose):
            
            ext_mat = np.eye(4)
            R = Rotation.as_matrix(Rotation.from_quat(cam_pose.R))
            ext_mat[:3, :3] = R
            ext_mat[:3, -1] = cam_pose.T
            return ext_mat
        
        def get_blocks(self, expand_threshold = 0.3):

            if (type(expand_threshold) is not float) and (type(expand_threshold) is not int):
                print("Wrong Expand Type. Should be Float")
                print(expand_threshold, type(expand_threshold))
                return 0 

            with open(self.st.point_pick_txt_path, "r") as f:
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
                
                quad  = self._expand_quad(quad, expand_threshold)

                quad_list.append(quad)

            self.st.quad_list = quad_list
            quad_path = os.path.join(self.st.cache_quad_path, "quad_list.npy")
            np.save(quad_path, quad_list)

        def make_folders(self):

            if self.st.quad_list is None:
                self.get_blocks()
            print("[ INFO ] Making Folders.")
            for index in range(len(self.st.quad_list)):

                output_folder_one = os.path.join(self.st.output_path, "group_" + str(index))

                output_img_folder = os.path.join(output_folder_one, "images")

                output_other_folder = os.path.join(output_folder_one, "sparse", "0")

                os.makedirs(output_img_folder, exist_ok=True)
                os.makedirs(output_other_folder, exist_ok=True)

        def crop_pcds(self, downsample_scale = 1.0):

            """
            Must Run AFTER self.make_folders()
            """
            if self.st.pcd is None:
                print("[ WARN ] No PointCloud File Found.")
                return 0

            if downsample_scale > 0 and downsample_scale < 1:
                print("Downsample pointcloud with scale %f" %downsample_scale)
                downsample_k = round(1 / downsample_scale)
                self.st.pcd = self.st.pcd.uniform_down_sample(every_k_points = downsample_k)

            print("[ INFO ] Cropping and Saving PointClouds")

            for index, quad in tqdm(enumerate(self.st.quad_list)):
                
                new_pcd = self._do_crop_with_quad(self.st.pcd, quad)

                output_folder_one = os.path.join(self.st.output_path, "group_" + str(index))
                output_pcd_path = os.path.join(output_folder_one, "sparse", "0", "points3D.ply")
                o3d.io.write_point_cloud(output_pcd_path, new_pcd)

        def crop_mesh_get_matrix2(self):
            
            if self.st.mesh is None:
                pass
                # TODO: 加上异常检测
            block_num = len(self.st.quad_list)
            triange_num = np.asarray((self.st.mesh.mesh.triangles)).shape[0]
            matrix_2 = np.zeros((block_num, triange_num), dtype=np.uint8)

            for index, quad in tqdm(enumerate(self.st.quad_list)):

                self.st.mesh.meshprocess.crop_mesh(quad = quad)

                cropped_index = self.st.mesh.meshprocess.cropped_index

                matrix_2[index, cropped_index] = 1

            matrix_2 = np.transpose(matrix_2, (1, 0))

            self.st.matrix_2 = matrix_2

            matrix_2_path = os.path.join(self.st.cache_matrix_path, "matrix_2.npy")
            np.save(matrix_2_path, matrix_2)

        def select_images(self, threshold):

            matrix_3 = np.zeros((len(self.st.cam_pose_list), self.st.matrix_2.shape[1]), dtype=np.uint32)

            for index in range(len(self.st.cam_pose_list)):

                hit_triangles_index = np.load(os.path.join(self.st.cache_matrix_path, "matrix_1_%03d.npy" %(index))).astype(np.uint32)
                matrix_1 = np.zeros((1, self.st.triangle_num), dtype=np.uint8)
                matrix_1[0, hit_triangles_index] = 1
                matrix_3[index, :] = np.matmul(matrix_1 , self.st.matrix_2.astype(np.uint32))

            self.st.matrix_3 = matrix_3

            matrix_3_path = os.path.join(self.st.cache_matrix_path, "matrix_3.npy")
            np.save(matrix_3_path, matrix_3)

            print("[ INFO ] Selecting Images.")

            matrix_4 = matrix_3 / np.sum(self.st.matrix_2.astype(np.uint32), axis=0).repeat(matrix_3.shape[0]).reshape(matrix_3.shape[1], -1).T

            matrix_5 = np.where(matrix_4 > threshold, 1, 0)

            self.st.matrix_5 = matrix_5

            matrix_4_path = os.path.join(self.st.cache_matrix_path, "matrix_4.npy")
            matrix_5_path = os.path.join(self.st.cache_matrix_path, "matrix_5.npy")

            np.save(matrix_4_path, matrix_4)
            np.save(matrix_5_path, matrix_5)

            for index in range(len(self.st.quad_list)):
                
                output_folder_one = os.path.join(self.st.output_path, "group_" + str(index))
                output_img_folder = os.path.join(output_folder_one, "images")

                os.makedirs(output_img_folder, exist_ok=True)

                new_cam_pose_list = []

                for img_index, img_path in enumerate(self.st.cam_pose_list):

                    img_path = os.path.join(self.st.cache_img_path, os.path.basename(self.st.cam_pose_list[img_index].img_path))
                    if matrix_5[img_index, index] == 1:
                        output_img_path = os.path.join(output_img_folder, os.path.basename(img_path))
                        shutil.copy(img_path, output_img_path)
                        new_cam_pose_list.append(self.st.cam_pose_list[img_index])

                self.save_cameras(new_cam_pose_list, index)
        
        def save_cameras(self, cam_pose_list, index):

            # save_intrinsics
            output_folder_one = os.path.join(self.st.output_path,"group_" + str(index))
            output_other_folder = os.path.join(output_folder_one, "sparse", "0")
            
            intrinsic_path = os.path.join(output_other_folder, "cameras.txt")
            colmap_intrinsic_text = ["1", "PINHOLE", self.st.new_width, self.st.new_height, self.st.fx, self.st.fy, self.st.cx, self.st.cy]
            with open(intrinsic_path, "w+") as file:
                for i in range(len(colmap_intrinsic_text)):
                    try:
                        file.write(colmap_intrinsic_text[i])
                    except:
                        file.write(str(colmap_intrinsic_text[i]))
                    file.write(" ")
                file.write("\n")
            print("[ INFO ] Group %d Intrinsic Saved in %s" % (index, intrinsic_path))

            # save_extrinsics

            extrinsic_path = os.path.join(output_other_folder, "images.txt")
            with open(extrinsic_path, "w+") as file:
                for idx, cam_pose in enumerate(cam_pose_list):
                    image_name = os.path.basename(cam_pose.img_path)
                    pose = convert_save_poses(cam_pose.T.tolist() + cam_pose.R.tolist())
                    output_line = f"{idx +1 } {pose[0]} {pose[1]} {pose[2]} {pose[3]} {pose[4]} {pose[5]} {pose[6]} 1 {image_name}\n No Content \n"
                    file.write(output_line)      
            
            print("[ INFO ] Group %d Extrinsic Saved in %s" % (index, extrinsic_path))



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