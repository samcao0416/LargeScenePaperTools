import argparse
import os
import open3d as o3d
import imageio
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

def get_parse():

    parser = argparse.ArgumentParser()
    parser.add_argument("--txt", type=str, help="image.txt, aka extrinsic path", required=True)
    parser.add_argument("--cam", type=str, help="cameras.txt, aka intrinsic path", required=True)
    parser.add_argument("--mesh", type=str, help="mesh path", required=True)
    parser.add_argument("--output", type=str, help="depth image output folder path", required=False, default=None)
    parser.add_argument("--format", type=str, choices=['16', '32'], default='16')
    args = parser.parse_args()
    return args

def generate_depth(vis, intrinsic, extrinsic):
    ctr = vis.get_view_control()

    camera_parameters = o3d.camera.PinholeCameraParameters()
    camera_parameters.intrinsic = intrinsic
    camera_parameters.extrinsic = extrinsic
    ctr.convert_from_pinhole_camera_parameters(camera_parameters, True)
    vis.poll_events()
    vis.update_renderer()
    depth_img = vis.capture_depth_float_buffer(True)
    return depth_img

def save_depth_img(img_name, format, depth_image, output_dir, debug = False):
    
    depth_image = np.asarray(depth_image) * 1000

    if format == "16":
        depth_image = depth_image.astype(np.uint16)

    elif format == "32":
        depth_image = depth_image.astype(np.uint32)
    
    else:
        raise ValueError("Unrecognized image format")
        
    imageio.imwrite(output_dir + "/" + img_name + ".png", depth_image)

    if debug:
        depth_vis = depth_image / 1000
        depth_vis /= np.max(depth_vis)
        depth_vis = depth_vis * 255
        depth_vis = depth_vis.astype(np.uint8)
        depth_vis_name = img_name + "_vis.png"
        imageio.imwrite(output_dir + "/" + depth_vis_name, depth_vis)

def read_intrinsics_text(path):
    
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                assert model == "PINHOLE", "While the loader support other types, the rest of the code assumes PINHOLE"
                width = int(elems[2])
                height = int(elems[3])
                fx = float(elems[4])
                fy = float(elems[5])
                cx = float(elems[6])
                cy = float(elems[7])
                
                K = np.array([[fx,  0, cx],
                              [ 0, fy, cy],
                              [ 0,  0,  1]])
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
                        width,  # 图像宽度
                        height,  # 图像高度
                        fx,  # x轴的焦距
                        fy,  # y轴的焦距
                        cx,  # x轴的主点
                        cy  # y轴的主点
                    )
    return intrinsic

def read_extrinsics_text(path):
    
    images = []
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9][:-4]
                try:
                    elems = fid.readline().split()
                    xys = np.column_stack([tuple(map(float, elems[0::3])),
                                        tuple(map(float, elems[1::3]))])
                    point3D_ids = np.array(tuple(map(int, elems[2::3])))
                except:
                    xys = None
                    point3D_ids = None

                pose = np.eye(4, 4)
                pose[:3, :3] = np.array(Rotation.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix())
                pose[:3, 3] = tvec
                images.append([pose, image_name])

    return images

if __name__=="__main__":
    args = get_parse()
    mesh = o3d.io.read_triangle_mesh(args.mesh)
    mesh.compute_vertex_normals()

    intrinsic = read_intrinsics_text(args.cam)
    cam_images = read_extrinsics_text(args.txt)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=intrinsic.width, height=intrinsic.height)
    vis.add_geometry(mesh)
    
    

    if args.output is None:
        output_folder = os.path.dirname(args.txt) + "/depths"
        
    else:
        output_folder = args.output
    
    os.makedirs(output_folder, exist_ok=True)

    for cam_image in tqdm(cam_images):
        
        image_name = cam_image[1]
        extrinsic = cam_image[0]
        depth_image = generate_depth(vis, intrinsic,extrinsic)
        save_depth_img(image_name, args.format, depth_image, output_folder,  False)
