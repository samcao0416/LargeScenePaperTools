import glob
import cv2
import numpy as np
import os
import math
import tqdm
import xml.etree.ElementTree as ET
from typing import NamedTuple

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

class undistort_cameras_agi():
    def __init__(self, path, ext = "JPG", workspace = None):
        image_folder = path
        self.image_path_list = sorted(glob.glob(os.path.join(image_folder,"*.%s" %(ext))))
        if workspace is None:
            self.workspace = os.path.join(os.path.dirname(image_folder), os.path.basename(image_folder) + "_READ_workspace")
        else:
            self.workspace = workspace
        os.makedirs(self.workspace, exist_ok=True)

    def change_format(self, metashape_path : str, agi_xml_path : str, img_path : str = None, output_path = None):
        # the precalibrated camrea format is Agisoft xml format, change it into opencv format
        if img_path is None:
            img_path = self.image_path_list[0]
        if output_path is None:
            output_path = os.path.join(self.workspace, "opencv_format.xml")
        os.system("%s -r ../AgiScripts/agi_change_fisheye_format_201.py --img %s --xml %s --out %s" %(metashape_path, img_path, agi_xml_path, output_path))

    def set_image_params_and_run(self, xml_path: str = None, width : int = 748, height : int = 748, fovX : float = 1.2, fovY : float = 0.0, write_imgs: bool = True):
        
        if xml_path is None:
            xml_path = os.path.join(self.workspace, "opencv_format.xml")
        
        cv_file = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_READ)
        image_width_raw = cv_file.getNode('image_Width').real()
        image_height_raw = cv_file.getNode('image_Height').real()
        camera_intrinsic = cv_file.getNode('Camera_Matrix').mat()
        camera_distortion = cv_file.getNode('Distortion_Coefficients').mat()
        D = np.array([camera_distortion[0][0], camera_distortion[1][0], camera_distortion[4][0], 0])
        
        scale_w = width / image_width_raw
        scale_h = height / image_height_raw

        if scale_w != scale_h:
            print("[ WARN ] Meet different width / height scale, use width_scale")
            scale = scale_w

        else:
            scale = scale_w

        

        self.new_width = int(image_width_raw * scale)
        self.new_height = int(image_height_raw * scale)

        
        fx = self.new_width / (2 * math.tan(fovX / 2))
        if fovY != 0.0:
            fy = self.new_height / (2 * math.tan(fovY / 2))
        else:
            fy = fx
        cx = self.new_width / 2 + scale * (camera_intrinsic[0][2] - image_width_raw / 2)
        cy = self.new_height / 2 + scale * (camera_intrinsic[1][2] - image_height_raw / 2)

        C_in = camera_intrinsic.copy()
        C_in[0,0] = fx
        C_in[1,1] = fy
        C_in[0,2] = cx
        C_in[1,2] = cy
        self.C_in = C_in

        print("[ INFO ] New image size: %d x %d" %(self.new_width, self.new_height))
        print("[ INFO ] New Camera Parameters: fx: %f, fy: %f, cx: %f, cy: %f" %(fx, fy, cx, cy))
        print("[ INFO ] Undistorting Images")

        if write_imgs:
            undistort_image_folder = os.path.join(self.workspace, "undistorted_images_%04dx%04d_%02.1f" % (self.new_width, self.new_height, fovX))
            os.makedirs(undistort_image_folder, exist_ok=True)
        else:
            undistort_images = []
        
        for image_path in tqdm.tqdm(self.image_path_list):

            img = cv2.imread(image_path)
            undistorted_image = cv2.fisheye.undistortImage(img, camera_intrinsic, D=D, Knew=C_in, new_size=(self.new_width, self.new_height))
            if write_imgs:
                cv2.imwrite(os.path.join(undistort_image_folder, os.path.basename(image_path)), undistorted_image)
            else:
                undistort_images.append(undistorted_image)

        if write_imgs:

            print("[ INFO ] %d images undistorted and saved to %s" %(len(self.image_path_list), undistort_image_folder))

        else:
            return undistort_images

    def get_opencv_xml_path(self):
        return os.path.join(self.workspace, "opencv_format.xml")
    

class undistort_cameras_opencv():
    def __init__(self, path = None, image_path_list = None, ext = "JPG", workspace = None, writing_mode : bool = True):
        if path is not None:
            image_folder = path
            self.image_path_list = sorted(glob.glob(os.path.join(image_folder,"*.%s" %(ext))))
        else:
            if image_path_list is not None:
                self.image_path_list = image_path_list
        self.writing_mode = writing_mode
        if self.writing_mode:
            if workspace is None:
                self.workspace = os.path.join(os.path.dirname(image_folder), os.path.basename(image_folder) + "_READ_workspace")
            else:
                self.workspace = workspace
            os.makedirs(self.workspace, exist_ok=True)

    def read_xml(self, xml_path : str = None):
        if xml_path is None or (not os.path.exists(xml_path)):
            raise FileExistsError("XML File does not exists.")
        xml_tree = ET.parse(xml_path)
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

        self.cam_int = Cam_Intrinsic_OpenCv(width=width,
                                       height=height,
                                       f = f,
                                       cx=cx,
                                       cy=cy,
                                       k1=k1,
                                       k2=k2,
                                       k3=k3,
                                       k4=k4)
    
    def set_image_params_and_run(self, width : int = 748, height : int = 748, fovX : float = 1.2, fovY : float = 0.0, fx = 0.0, fy = 0.0):

        image_width_raw = self.cam_int.width
        image_height_raw = self.cam_int.height
        camera_intrinsic = np.array([[self.cam_int.f, 0, image_width_raw / 2 + self.cam_int.cx],
                                     [0, self.cam_int.f, image_height_raw / 2 + self.cam_int.cy],
                                     [0,0,1]])
        cam_distortion = np.array([self.cam_int.k1,
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

        

        self.new_width = int(image_width_raw * scale)
        self.new_height = int(image_height_raw * scale)

        if fx == 0.0 and fy == 0.0:            
            fx = self.new_width / (2 * math.tan(fovX / 2))
            if fovY != 0.0:
                fy = self.new_height / (2 * math.tan(fovY / 2))
            else:
                fy = fx
        elif fx != 0.0 and fy == 0.0:
            fy = fx
        elif fy != 0.0 and fx == 0.0:
            fx = fy
        
                
        cx = self.new_width / 2 + scale * (camera_intrinsic[0][2] - image_width_raw / 2)
        cy = self.new_height / 2 + scale * (camera_intrinsic[1][2] - image_height_raw / 2)

        C_in = camera_intrinsic.copy()
        C_in[0,0] = fx
        C_in[1,1] = fy
        C_in[0,2] = cx
        C_in[1,2] = cy
        self.C_in = C_in

        print("[ INFO ] New image size: %d x %d" %(self.new_width, self.new_height))
        print("[ INFO ] New Camera Parameters: fx: %f, fy: %f, cx: %f, cy: %f" %(fx, fy, cx, cy))
        print("[ INFO ] Undistorting Images")

        if self.writing_mode:
            undistort_image_folder = os.path.join(self.workspace, "undistorted_images_%04dx%04d_%02.1f" % (self.new_width, self.new_height, fovX))
            os.makedirs(undistort_image_folder, exist_ok=True)
        else:
            undistort_images = []
        
        for image_path in tqdm.tqdm(self.image_path_list):

            img = cv2.imread(image_path)
            undistorted_image = cv2.fisheye.undistortImage(img, camera_intrinsic, D=cam_distortion, Knew=C_in, new_size=(self.new_width, self.new_height))
            if self.writing_mode:
                cv2.imwrite(os.path.join(undistort_image_folder, os.path.basename(image_path)), undistorted_image)
            else:
                undistort_images.append(undistorted_image)

        if self.writing_mode:

            print("[ INFO ] %d images undistorted and saved to %s" %(len(self.image_path_list), undistort_image_folder))

        else:
            return undistort_images

    def get_focal(self):
        return self.C_in[0,0], self.C_in[1, 1]
    
    def get_cxcy(self):
        return self.C_in[0, 2], self.C_in[1, 2]

    def get_shape(self):
        return self.new_width, self.new_height
    
class undistort_single_cam_opencv():
    def __init__(self, workspace = None, data_type = 1):
        if workspace is None:
            self.workspace = os.path.join("./undistorted_images")
        else:
            self.workspace = workspace

        self.data_type = data_type

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

    def read_xml(self, K_xml_path: str = None):

        if self.data_type:
            if K_xml_path is None or (not os.path.exists(K_xml_path)):
                raise FileExistsError("XML File does not exists.")
            self.cam_int = self._read_K_xml(K_xml_path)
        else:
            if K_xml_path is None or (not os.path.exists(K_xml_path[0])):
                raise FileExistsError("XML File does not exists.")
            
            self.cam_int = []
            
            for K_xml_path_single in K_xml_path:
                self.cam_int.append(self._read_K_xml(K_xml_path_single))
                

    def set_param(self, width: int = 1500, height:int = 1500, fovX: float = 1.2, fovY: float = None, fx: float = None, fy: float = None, cx: float = None, cy: float = None):
        

        if self.cam_int is None:
            raise SystemError("No camera intrinsic parameters loaded. Please run .read_xml() first.")
        
        

        if self.data_type:
            image_width_raw = self.cam_int.width
            image_height_raw = self.cam_int.height

            self.camera_intrinsic = np.array([[self.cam_int.f, 0, image_width_raw / 2 + self.cam_int.cx],
                                        [0, self.cam_int.f, image_height_raw / 2 + self.cam_int.cy],
                                        [0,0,1]])
            self.cam_distortion = np.array([self.cam_int.k1,
                                    self.cam_int.k2,
                                    self.cam_int.k3,
                                    self.cam_int.k4])
        else:
            image_width_raw = self.cam_int[0].width
            image_height_raw = self.cam_int[0].height

            self.camera_intrinsic = []
            self.cam_distortion = []
            for cam_int_single in self.cam_int:
                self.camera_intrinsic.append(np.array([[cam_int_single.f, 0, image_width_raw / 2 + cam_int_single.cx],
                                        [0, cam_int_single.f, image_height_raw / 2 + cam_int_single.cy],
                                        [0,0,1]]))
                self.cam_distortion.append(np.array([cam_int_single.k1,
                                    cam_int_single.k2,
                                    cam_int_single.k3,
                                    cam_int_single.k4]))
        
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



    def undistort_image(self, cam_pose, output_path:str = None, write_mode = True):
        
        if self.C_in is None:
            raise SystemError("Parameters not set. Please run .set_param() first.")
        
        img_path = cam_pose.img_path
        img = cv2.imread(img_path)

        if self.data_type:
            try:
                undistorted_image = cv2.fisheye.undistortImage(img, self.camera_intrinsic, D=self.cam_distortion, Knew=self.C_in, new_size=(self.new_width, self.new_height))
            except:
                print("[ ERROR ] Undistort image failed, img path is %s" %(img_path))
                undistorted_image = None
        else:
            index = cam_pose.titan_index - 1
            undistorted_image = cv2.fisheye.undistortImage(img, self.camera_intrinsic[index], D=self.cam_distortion[index], Knew=self.C_in, new_size=(self.new_width, self.new_height))
        


        if write_mode:
            if output_path is None:
                output_path = os.path.join(self.workspace, os.path.basename(img_path))

            if undistorted_image is not None:
                cv2.imwrite(output_path, undistorted_image)
        else:
            return undistorted_image

    def get_focal(self):
        return self.C_in[0,0], self.C_in[1,1]
    
    def get_cxcy(self):
        return self.C_in[0, 2], self.C_in[1, 2]

    def get_shape(self):
        return self.new_width, self.new_height
    
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