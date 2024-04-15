from xml.dom import minidom
from xml.etree import ElementTree as ET
import numpy as np
import os
import argparse

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-xml_1", "--xml_1", type=str)
    parser.add_argument("-xml_2", "--xml_2", type=str)
    parser.add_argument("-out", "--output", type=str, default=None)

    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = argparser()

    xml_1 = args.xml_1
    xml_2 = args.xml_2

    # read_xml2
    tree = ET.parse(xml_2)
    root = tree.getroot()
    photo_xml = root.find("Block").find("Photogroups").find("Photogroup")

    pose_list = []

    for ele in photo_xml:
        name = ele.tag
        if name == "Photo":
            rotation_node = ele.find('Pose').find('Rotation')
            translation_node = ele.find('Pose').find('Center')

            pose = []
            for i in range(3):
                for j in range(3):
                    index = "M_" + str(i) + str(j)
                    pose.append(float(rotation_node.find(index).text))
            
            pose.append(float(translation_node.find('x').text))
            pose.append(float(translation_node.find('y').text))
            pose.append(float(translation_node.find('z').text))

            img_name = os.path.basename(ele.find("ImagePath").text)
            pose_list.append([pose, img_name])
    
    # add to xml_1
    tree = ET.parse(xml_1)
    root = tree.getroot()
    photo_xml = root.find("Block").find("Photogroups").find("Photogroup")

    photos_1 = photo_xml.findall("Photo")
    current_id = len(photos_1)

    doc = minidom.parse(xml_1)
    photogroups = doc.getElementsByTagName('Photogroup')
    last_photogroup = photogroups[-1]

    for pose in pose_list:
        Photo = doc.createElement('Photo')
        last_photogroup.appendChild(Photo)

        Idx = doc.createElement('Id')
        Photo.appendChild(Idx)
        Idx.appendChild(doc.createTextNode(str(current_id)))

        ImagePath = doc.createElement('ImagePath')
        Photo.appendChild(ImagePath)
        path_text = doc.createTextNode(pose[1])
        ImagePath.appendChild(path_text)

        Component = doc.createElement('Component')
        Photo.appendChild(Component)
        component_idx = doc.createTextNode("1")
        Component.appendChild(component_idx)

        Pose = doc.createElement('Pose')
        Photo.appendChild(Pose)
        Rotation = doc.createElement('Rotation')
        Pose.appendChild(Rotation)
        for i in range(3):
            for j in range(3):
                node = "M_" + str(i) + str(j)
                rotate_node = doc.createElement(node)
                rotate_node.appendChild(doc.createTextNode(str(pose[0][i+j])))
                Rotation.appendChild(rotate_node)

        Center = doc.createElement('Center')
        Pose.appendChild(Center)
        cnter_label = ['x', 'y', 'z']
        for i in range(3):
            c_node = doc.createElement(cnter_label[i])
            c_node.appendChild(doc.createTextNode(str(pose[0][9+i])))
            Center.appendChild(c_node)

        current_id += 1

    if args.output is None:
        output = os.path.join(os.path.dirname(args.xml_1), "pose_output.xml")
    else:
        output = args.output
        os.makedirs(os.path.dirname(output), exist_ok=True)

    with open(output, "w", encoding='UTF-8') as fh:
        doc.writexml(fh,indent='',addindent='\t',newl='\n',encoding='UTF-8')