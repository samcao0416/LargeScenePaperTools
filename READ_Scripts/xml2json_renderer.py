import xml.etree.ElementTree as ET
import json
import os
import argparse
import numpy as np


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, required=True, help = "path to input .xml file")
    parser.add_argument("--pcd", type=str, default=None)
    parser.add_argument("-net", "--net_ckpt", type=str, default=None)
    parser.add_argument("-tex", "--tex_ckpt", type=str, default=None)
    parser.add_argument("-yml", "--view_yaml", type=str, default=None)
    parser.add_argument("--out", type=str, default=None, help = "output .json path")
    args = parser.parse_args()

    return args

def intrinsics_from_xml(xml_file):
    root = ET.parse(xml_file).getroot()
    calibration = root.find('chunk/sensors/sensor/calibration')
    resolution = calibration.find('resolution')
    width = float(resolution.get('width'))
    height = float(resolution.get('height'))
    f = float(calibration.find('f').text)
    cx = width/2
    cy = height/2

    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0,  1]
        ], dtype=np.float32)

    return K, (width, height)

def read_xml(xml_file):
    transforms = []
    tree = ET.parse(xml_file)
    root = tree.getroot()
    translation = np.array([float(x) for x in root.findall('chunk/components/component')[0].findall('transform/translation')[0].text.split()])
    child = root.findall('chunk/cameras')[0][0]
    if child.tag == "camera":
        for e in root.findall('chunk/cameras')[0].findall("camera"):
            label = e.get("label")
            transform = e.find("transform").text.split(' ')
            transform = np.array([float(x) for x in transform])
            transform = transform.reshape(4, 4)
            transform[0:3, 3] += translation

            transforms.append([label, transform])

    elif child.tag == "group":
        for g in root.findall('chunk/cameras')[0].findall("group"):
            for e in g.findall("camera"):
                label = e.get('label')
                transform = e.find("transform").text.split(' ')
                transform = np.array([float(x) for x in transform])
                transform.reshape(4, 4)
                transform[0:3, 3] += translation

                transforms.append([label, transform])

    return transforms

def read_yaml(yml_file):
    import yaml
    with open(yml_file, 'r') as file:

        data = yaml.load(file, Loader=yaml.FullLoader)

        # 现在data就是一个Python字典，包含了YAML文件中的所有数据
        scene_data = {}
        scene_data["pointcloud"] = data.get("pointcloud")
        scene_data["net_ckpt"] = data.get("net_ckpt")
        scene_data["texture_ckpt"] = data.get("texture_ckpt")
    
    return scene_data

def write_json(scene_data, out_path):
    key_frames_json = {
            "pointcloud": scene_data['pointcloud'],
            "net_ckpt": scene_data['net_ckpt'],
            "texture_ckpt": scene_data['texture_ckpt'],
            "w": np.float(scene_data["viewport_size"][0]),
            "h":   np.float(scene_data["viewport_size"][1]),
            "f_x": np.float(scene_data['intrinsic_matrix'][0][0]),
            "f_y": np.float(scene_data['intrinsic_matrix'][1][1]),
            "c_x": np.float(scene_data['intrinsic_matrix'][0][2]),
            "c_y": np.float(scene_data['intrinsic_matrix'][1][2]),
            "key_frames":[]
        }
    
    for transform in scene_data["transforms"]:
        image_name = transform[0]
        extrinsic = transform[1]
        

if __name__ == "__main__":

    args = argparser()

    if (args.pcd is None or \
       args.net_ckpt is None or \
       args.tex_ckpt is None) and \
       args.view_yaml is None:
        raise RuntimeError("You should provide pcd, net_ckpt and tex_ckpt path or view yaml path")

    scene_data = {}

    if args.view_yaml is None:
        scene_data["pointcloud"] = args.pcd
        scene_data["net_ckpt"] = args.net_ckpt
        scene_data["texture_ckpt"] = args.tex_ckpt
    else:
        scene_data = read_yaml(args.view_yaml)
        pass

    K, (width, height) = intrinsics_from_xml(args.xml)

    

    scene_data['intrinsic_matrix'] = K

    scene_data["viewport_size"] = [width, height]


    # print(K, width, height)

    transforms = read_xml(args.xml)

    scene_data["transforms"] = transforms

