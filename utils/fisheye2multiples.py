import numpy as np
import open3d as o3d
import os
import math
import cv2
import sys
import glob
import xml.etree.ElementTree as ET
from typing import NamedTuple
from scipy.spatial.transform import Rotation


class Cam_Intrinsic_OpenCv(NamedTuple):
    width: int
    height: int
    f: float
    cx: float
    cy: float
    k1: float
    k2: float
    k3: float
    k4: float


class Fisheye2Multiples():

    def __init__(self, workspace:str = None, data_type: int = 1, rad = np.pi / 4):
        if workspace is None:
            self.workspace = os.path.join("./fisheye2multiple_images")
        else:
            self.workspace = workspace

        os.makedirs(self.workspace, exist_ok=True)

        if data_type == 0:
            raise TypeError("Titan is not supported now.")

        self.cam_int = None

        self.new_width = None
        self.new_height = None
        self.fovX = None
        self.fovY = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.camera_intrinsic = None
        self.cam_distortion = None
        self.C_in = None

        self.rotation = None
        self._init_rotations(rad = rad)

    def read_xml(self, K_xml_path: str = None):
            
        if K_xml_path is None or (not os.path.exists(K_xml_path)):
            raise FileExistsError("XML File does not exists.")
        self.cam_int = self._read_K_xml(K_xml_path)

    
    def set_param(self, width: int = 1500, height:int = 1500, fovX: float = 1.2, fovY: float = None, fx: float = None, fy: float = None, cx: float = None, cy: float = None):

        if self.cam_int is None:
            raise SystemError("No camera intrinsic parameters loaded. Please run .read_xml() first.")
        
        image_width_raw = self.cam_int.width
        image_height_raw = self.cam_int.height

        self.camera_intrinsic = np.array([[self.cam_int.f, 0, image_width_raw / 2 + self.cam_int.cx],
                                        [0, self.cam_int.f, image_height_raw / 2 + self.cam_int.cy],
                                        [0,0,1]])
        self.cam_distortion = np.array([self.cam_int.k1,
                                self.cam_int.k2,
                                self.cam_int.k3,
                                self.cam_int.k4])
        
        scale_w = width / image_width_raw
        scale_h = height / image_height_raw

        if scale_w != scale_h:
            print("[ WARN ] Meet different width / height scale, use width_scale")
            scale = scale_w

        else:
            scale = scale_w

        new_width = int(image_width_raw * scale)
        new_height = int(image_height_raw * scale)

        if fx is None and fy is None:
            fx = new_width / (2 * math.tan(fovX / 2))
            if fovY is not None:
                fy = new_height / (2 * math.tan(fovY / 2))
            else:
                fy = fx
        
        elif fx is not None and fy is None:
            fy = fx
        elif fy is not None and fy is None:
            fx = fy

        if cx is None:
            cx = new_width / 2 # + scale * (self.camera_intrinsic[0][2] - image_width_raw / 2)

        if cy is None:
            cy = new_height / 2 # + scale * (self.camera_intrinsic[1][2] - image_height_raw / 2)

        self.new_width = new_width
        self.new_height = new_height
        self.fovX = fovX
        self.fovY = fovY
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        self.C_in = np.array([[self.fx, 0, self.cx],
                        [0, self.fy, self.cy],
                        [0, 0 , 1]])

        print("[ INFO ] New image size: %d x %d" %(self.new_width, self.new_height))
        print("[ INFO ] New Camera Parameters: fx: %f, fy: %f, cx: %f, cy: %f" %(fx, fy, cx, cy))

    
    def fisheye_to_multiples(self, cam_pose, views:list = ["F", "L","R", "U", "D"], output_folder:str = None, write_mode: bool = True):

        if self.C_in is None:
            raise SystemError("Parameters not set. Please run .set_param() first.")

        img_path = cam_pose.img_path

        new_cam_pose_list = []
        if not write_mode:
            new_img_list = []

        for view_id in views:

            new_cam_pose = self._convert_new_extrinsic(cam_pose=cam_pose, view_id=view_id)
            new_cam_pose_list.append(new_cam_pose)

        self.new_cam_pose_list = new_cam_pose_list

        points = self._generate_points_with_raycasting(views=views)

        fisheye_img = cv2.imread(img_path)

        rvecs = np.zeros((3, 1))
        tvecs = np.zeros((3, 1))

        for i, new_cam_pose in enumerate(self.new_cam_pose_list):

            pointsreshape  = points[i].reshape(-1, 3)

            pointsreshape = pointsreshape.reshape(-1, 1, 3)

            imagePoints, _ = cv2.fisheye.projectPoints(pointsreshape, rvecs, tvecs, self.camera_intrinsic, self.cam_distortion)

            imagePoints = imagePoints.reshape(self.new_height, self.new_width, 2) # TODO: height first or width first?

            new_img = fisheye_img[np.round(imagePoints[..., 1]).astype(int), np.round(imagePoints[..., 0]).astype(int)]

            if write_mode:
                if output_folder is None:
                    output_path = os.path.join(self.workspace, os.path.basename(new_cam_pose.img_path))

                else:
                    os.makedirs(output_folder, exist_ok=True)
                    output_path = os.path.join(output_folder, os.path.basename(new_cam_pose.img_path))

                cv2.imwrite(output_path, new_img)

            else:
                new_img_list.append(new_img)

        if not write_mode:
            return  new_img_list

    def get_focal(self):
        return self.C_in[0,0], self.C_in[1,1]
    
    def get_cxcy(self):
        return self.C_in[0, 2], self.C_in[1, 2]

    def get_shape(self):
        return self.new_width, self.new_height
    
    def get_new_cam_pose_list(self):
        return self.new_cam_pose_list
    
    def get_new_view_num(self):
        return len(self.new_cam_pose_list)
        

    

    def _convert_new_extrinsic(self, cam_pose, view_id: str):
        
        new_matrix = np.eye(4)
        new_rotation = Rotation.as_matrix(Rotation.from_quat( cam_pose.R)) @ np.linalg.inv(self.rotations_for_colmap[view_id])
        new_matrix[:3, :3] = new_rotation
        new_matrix[:3, -1] = cam_pose.T

        new_R = Rotation.as_quat(Rotation.from_matrix(new_rotation))

        new_img_path = os.path.join(os.path.dirname(cam_pose.img_path), os.path.basename(cam_pose.img_path)[:-4] + "_" + view_id + ".jpg") 
        new_cam_pose = self.CamPoseCol(titan_index=cam_pose.titan_index,
                                       img_path = new_img_path,
                                       R = new_R,
                                       T = cam_pose.T)
        
        return new_cam_pose
        


    def _generate_points_with_raycasting(self, views):

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius = 10)
        sphere = sphere.subdivide_midpoint(number_of_iterations = 4)

        scene = o3d.t.geometry.RaycastingScene()
        mesh_for_raycast = o3d.t.geometry.TriangleMesh.from_legacy(sphere)

        scene.add_triangles(mesh_for_raycast)

        points = []

        # 使用相机的内外参去创建点
        intrinsic_matrix = o3d.core.Tensor(self.C_in, dtype = o3d.core.float64)

        for view_id in views:

            extrinsic = np.eye(4)
            extrinsic[:3, :3] = self.rotations[view_id] # changed, figure out whether will works

            extrinsic_matrix = o3d.core.Tensor(np.linalg.inv(extrinsic), dtype = o3d.core.float64)

            rays = scene.create_rays_pinhole(intrinsic_matrix, extrinsic_matrix, self.new_width, self.new_height)

            result = scene.cast_rays(rays)

            result["t_hit"] = result["t_hit"].numpy()
            
            rays = rays.numpy()
            
            point = rays[..., 3:] * result['t_hit'][..., None]

            points.append(point)

        return points
            
            

    def _read_K_xml(self, K_xml_path):
        xml_tree = ET.parse(K_xml_path)
        cal_tree = xml_tree.getroot()
        
        width = int(cal_tree.find("width").text)
        height = int(cal_tree.find("height").text)
        f  = float(cal_tree.find("f").text)
        cx = float(cal_tree.find("cx").text)
        cy = float(cal_tree.find("cy").text)
        k1 = float(cal_tree.find("k1").text)
        k2 = float(cal_tree.find("k2").text)
        k3 = float(cal_tree.find("k3").text)
        try:
            k4 = float(cal_tree.find("k4").text)
        except:
            k4 = 0.0

        return Cam_Intrinsic_OpenCv(width=width,
                                       height=height,
                                       f = f,
                                       cx=cx,
                                       cy=cy,
                                       k1=k1,
                                       k2=k2,
                                       k3=k3,
                                       k4=k4)
    
    def _init_rotations(self, rad = np.pi / 4): 
        rotation_front = o3d.geometry.get_rotation_matrix_from_xyz((0,    0, 0))    # 上下左右互换了一下
        rotation_left  = o3d.geometry.get_rotation_matrix_from_xyz((0, -rad, 0))    # 上下左右互换了一下
        rotation_right = o3d.geometry.get_rotation_matrix_from_xyz((0,  rad, 0))    # 上下左右互换了一下
        rotation_down  = o3d.geometry.get_rotation_matrix_from_xyz((-rad, 0, 0))    # 上下左右互换了一下
        rotation_up    = o3d.geometry.get_rotation_matrix_from_xyz(( rad, 0, 0))    # 上下左右互换了一下

        self.rotations = {
            "F": rotation_front,
            "L": rotation_left,
            "R": rotation_right,
            "U": rotation_up,
            "D": rotation_down
        }

        self.rotations_for_colmap = {
            "F": rotation_front,
            "L": rotation_right,
            "R": rotation_left,
            "U": rotation_down,
            "D": rotation_up
        }

    class CamPoseCol(NamedTuple):
            titan_index: int
            img_path: str
            R: np.ndarray
            T: np.ndarray


    class DeBug():
        def __init__(self, outer_class):
            self.st = outer_class

        