import argparse
import os
import open3d as o3d
import imageio
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import torch
from PIL import Image

def get_parse():

    parser = argparse.ArgumentParser()
    parser.add_argument("--txt", type=str, help="image.txt, aka extrinsic path", required=True)
    parser.add_argument("--cam", type=str, help="cameras.txt, aka intrinsic path", required=True)
    parser.add_argument("--mesh", type=str, help="mesh path", required=True)
    parser.add_argument("--output", type=str, help="depth image output folder path", required=False, default=None)
    parser.add_argument("--format", type=str, choices=['8','16', 'float'], default='float')
    parser.add_argument("--debug", action="store_true")
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
    
    depth_img = vis.capture_depth_float_buffer(True) # TODO: No inf number exception control
    colored_img = vis.capture_screen_float_buffer(False)
    return colored_img, depth_img

def save_depth_img(img_name, format, depth_image, output_dir, debug = False):
    
    depth_image = np.asarray(depth_image)
    if debug:
        depth_vis = depth_image.copy()

    if format == "16":
        # depth_image /= np.max(depth_image)
        
        depth_image = torch.from_numpy(depth_image)
        depth_image = torch.where(depth_image < 10.0, depth_image / 10.0, 2.0 - 10 / depth_image) / 2.0

        depth_image = depth_image.numpy() * 65535
        depth_image = depth_image.astype(np.uint16)
        
        imageio.imwrite(output_dir + "/" + img_name + ".png", depth_image)

    elif format == "8":
        # depth_image /= np.max(depth_image)
        depth_image = torch.from_numpy(depth_image)
        depth_image = torch.where(depth_image < 10.0, depth_image / 10.0, 2.0 - 10 / depth_image) / 2.0
        depth_image = depth_image.numpy() * 255
        depth_image = depth_image.astype(np.uint8)
        imageio.imwrite(output_dir + "/" + img_name + ".png", depth_image)
    
    elif format == "float":
        # depth_image = depth_image.astype(np.float32) # TODO: find the difference reason
        depth_image = torch.from_numpy(depth_image)
        depth_image = torch.where(depth_image < 10.0, depth_image / 10.0, 2.0 - 10 / depth_image) / 2.0
        np.save(output_dir + "/" + img_name + ".npy", depth_image)
    
    else:
        raise ValueError("Unrecognized image format")
        
    

    if debug:
        # depth_vis /= np.max(depth_vis)
        # depth_vis = depth_vis * 255
        depth_vis = torch.from_numpy(depth_vis)
        depth_vis = torch.where(depth_vis < 10.0, depth_vis / 10.0, 2.0 - 10 / depth_vis) / 2.0
        depth_vis = depth_vis.numpy() * 255
        depth_vis = depth_vis.astype(np.uint8)
        depth_vis_dir = output_dir + "_debug"
        os.makedirs(depth_vis_dir, exist_ok=True)
        depth_vis_name = img_name + "_vis.png"
        imageio.imwrite(depth_vis_dir + "/" + depth_vis_name, depth_vis)

def save_colored_img(img_name, colored_image, output_dir, debug = False):
    colored_image = np.asarray(colored_image) * 255.0
    colored_image = colored_image.astype(np.uint8)
    imageio.imwrite(output_dir + "/" + img_name + ".jpg", colored_image)

def save_depth_img_with_plane(img_name, depth_image, colored_image, output_dir, debug = False):

    depth_image = np.asarray(depth_image)
    colored_image = np.asarray(colored_image)
    # colored_image = colored_image.astype(np.uint8)
    colored_image = torch.from_numpy(colored_image)
    depth_image = torch.from_numpy(depth_image)
    depth_image = torch.where(depth_image < 10.0, depth_image / 10.0, 2.0 - 10 / depth_image) / 2.0
    # depth_image = depth_image.numpy() * 255
    # depth_image = depth_image.astype(np.uint8)
    

    depth_with_plane = Image.fromarray(torch.stack([depth_image * 255.0, colored_image[:,:,0] * 255.0], dim=2).detach().cpu().numpy().astype(np.uint8), mode="LA")
    depth_with_plane.save(output_dir + "/" + img_name + ".png")



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

    render_option = vis.get_render_option()
    render_option.light_on = True
    
    

    if args.output is None:
        output_folder = os.path.dirname(args.txt) + "/depths"
        output_colored_folder = os.path.dirname(args.txt) + "/depths_mask"
        
    else:
        output_folder = args.output
        output_colored_folder = os.path.dirname(args.output) + "/depths_mask"
    
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_colored_folder, exist_ok=True)

    for cam_image in tqdm(cam_images):
        
        image_name = cam_image[1]
        extrinsic = cam_image[0]
        colored_image, depth_image = generate_depth(vis, intrinsic,extrinsic)
        # save_depth_img(image_name, args.format, depth_image, output_folder,  args.debug)
        save_colored_img(image_name, colored_image, output_colored_folder, args.debug)
        save_depth_img_with_plane(image_name, depth_image, colored_image, output_folder, args.debug)
