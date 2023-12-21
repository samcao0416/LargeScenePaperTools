# +D 2023.9.17

from colmap_loader import read_extrinsics_text, read_intrinsics_text
import numpy as np
import open3d as o3d
import time
import trimesh
import os 
import cv2

# parameters
ply_file = "./input/sparse/0/points3D.ply"
extrinsics_file = "./input/sparse/0/images.txt"
intrinsics_file = "./input/sparse/0/cameras.txt"
mesh_file = "./input/sparse/0/mesh.ply" # NOTE: first, convert .obj to .ply

# ------ main ------
poses = read_extrinsics_text(extrinsics_file)
# print(poses)
intrinsics = read_intrinsics_text(intrinsics_file)
assert len(intrinsics) == 1 # NOTE: only 1 camera
# print(intrinsics[1].width, intrinsics[1].height)
view_width, view_height = intrinsics[1].width, intrinsics[1].height
intrinsics = np.array([[intrinsics[1].params[0], 0, intrinsics[1].params[2]],
                       [0, intrinsics[1].params[1], intrinsics[1].params[3]],
                       [0, 0, 1]])
# change the resolution
view_width, view_height = view_width // 3, view_height // 3
intrinsics = np.array([[intrinsics[0, 0] // 3, 0, intrinsics[0, 2] // 3],
                       [0, intrinsics[1, 1] // 3, intrinsics[1, 2] // 3],
                       [0, 0, 1]])

# print(intrinsics)
intrinsics = o3d.camera.PinholeCameraIntrinsic(view_width, view_height, 
        intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2])

poses_matrix = {} # c2w
name_list = {}
for k in poses:
    v = poses[k]
    
    pose_matrix = np.eye(4) # init   
    pose_matrix[:3, :3] = v.qvec2rotmat()
    pose_matrix[:3, 3] = np.array(v.tvec)
    
    poses_matrix[k] = pose_matrix
    name_list[k] = v.name
assert len(poses_matrix) == len(name_list)
    
# use open3d to read ply_file
# pcd = o3d.io.read_point_cloud(ply_file)

mesh = o3d.io.read_triangle_mesh(mesh_file)
vertices = np.asarray(mesh.vertices)
triangles = np.asarray(mesh.triangles)

# # use open3d to visualize the point cloud / mesh and poses
# vis = o3d.visualization.Visualizer()
# vis.create_window()

# # vis.add_geometry(pcd)
# vis.add_geometry(mesh)

# for k in poses_matrix:
#     v = poses_matrix[k]
#     v = np.linalg.inv(v) # w2c
#     vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=v[:3, 3]))
# vis.run()

# mesh = trimesh.load(mesh_file)
# mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)

# render images
vis = o3d.visualization.Visualizer()
# view_height, view_width = 1200, 1200
vis.create_window(width = view_width, height = view_height)
vis.add_geometry(mesh)

# check ./output/depth is exist
if not os.path.exists("./output/depth"):
    os.makedirs("./output/depth")
# check ./output/pcd is exist
if not os.path.exists("./output/pcd"):
    os.makedirs("./output/pcd")

for i in poses_matrix:
    camera = o3d.camera.PinholeCameraParameters()
    camera.intrinsic = intrinsics
    camera.extrinsic = poses_matrix[i]
    
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(camera, True)
    vis.poll_events()
    vis.update_renderer()
    
    # output depth images
    depth = vis.capture_depth_float_buffer(True)
    depth = np.asarray(depth)
    f_name =  "./output/depth/{:s}.png".format(name_list[i][:-4])
    res = cv2.imwrite(f_name, depth.astype(np.float32))
    assert res  
    
    # output point cloud with global poses
    depth = vis.capture_depth_float_buffer(True)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth, intrinsics, 
                                                          poses_matrix[i])
    o3d.io.write_point_cloud("./output/pcd/{:s}.ply".format(name_list[i][:-4]), pcd)
    

    time.sleep(0.2)

vis.destroy_window()
    
    



