2023/11/12

# Version 5: 
## Overall Functions:
* load_xml_transform(node: xml element): -> camera poses, 4 x 4 np.array \
   load xml data in blocks_exchange format data and return camera poses
* convert_save_poses(pose: 4 x 4 list): -> colmap format pose in quaternion format, 7 x 1 np.array \
   convert pose from 4 x 4 list to 7 x 1 np.array and save it in colmap format
* IndexRange: NamedTuple Class \
   a class to store the range of given image start, end, and step index

##  Agi2Colmap Class: 
### 1. function: initialize:
xml_path: path of extrinsics xml file in blocks_exchange format
img_folder: path of image folder
ply_path: path of pointcloud file in .ply format
output_path: path of output folder
mesh_path: path of mesh in .obj format, to generate mask, select images with. Not necesary
data_type: type of data in str, "insta" or "titan". \
&emsp;&emsp;&emsp;&emsp;&ensp; "insta": 1 \
&emsp;&emsp;&emsp;&emsp;&ensp; "titan": 0
### 2. class Check_Files:
#### 2.1 function: initialize:
self.oc: outerclass
#### 2.2 function run():



  