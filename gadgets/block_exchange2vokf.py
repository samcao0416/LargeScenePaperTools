import glob
import argparse
import os
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation
import numpy as np

def argparser():

    parser = argparse.ArgumentParser()
    parser.add_argument("-xml", "--xml", type=str, required=True, help = "path to input .xml file")
    parser.add_argument("-txt", "--txt", type=str, default=None)
    args = parser.parse_args()

    return args

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

if __name__=="__main__":
    args = argparser()
    
    xml_file = args.xml
    txt_file = args.txt

    if txt_file is None:
        txt_file = os.path.dirname(xml_file) + "/vokf.txt"

    tree = ET.parse(xml_file)
    root = tree.getroot()

    photo_xml = root.find("Block").find("Photogroups").find("Photogroup")

    pose_list = []

    for ele in photo_xml:
        name = ele.tag
        if name == "Photo":
            pose = load_xml_transform(ele.find("Pose"))
            img_name = os.path.basename(ele.find("ImagePath").text)
            pose_list.append([pose, img_name])

    with open(txt_file, "w+") as f:

        for poseNname in pose_list:
            rotate = poseNname[0][0:3, 0:3]
            R = Rotation.as_quat(Rotation.from_matrix(rotate))
            T = poseNname[0][0:3, 3]

            name = poseNname[1]

            output_line = f"{T[0]} {T[1]} {T[2]} {R[0]} {R[1]} {R[2]} {R[3]} {name}\n"

            f.write(output_line)

    print("Vokf.txt saved to %s" %(txt_file))
        

