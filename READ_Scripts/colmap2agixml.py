import argparse
import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial.transform import  Rotation 

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-txt", "--colmap_path", type=str)
    parser.add_argument("-out", "--output_path", type=str)
    parser.add_argument("-xml", "--xml_path", type = str)
    args = parser.parse_args()

    return  args

def copy_xml_heads(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for e in root.findall('chunk/cameras')[0].findall("camera"):
        root.findall('chunk/cameras')[0].remove(e)

    return tree

def copy_colmap_cams(xml_tree, colmap_path):
    root = xml_tree.getroot()
    translation = np.array([float(x) for x in root.findall('chunk/components/component')[0].findall('transform/translation')[0].text.split()])
    
    with open(colmap_path, "r") as f_in:
        while(True):

            line = f_in.readline()

            if not line:
                break

            line_ele = line.strip()

            if len(line_ele) > 0 and line[0] != "#":

                elems = line_ele.split()

                image_id = int(elems[0])
                
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                try:
                    elems = f_in.readline().split()
                    xys = np.column_stack([tuple(map(float, elems[0::3])),
                                        tuple(map(float, elems[1::3]))])
                    point3D_ids = np.array(tuple(map(int, elems[2::3])))
                except:
                    xys = None
                    point3D_ids = None

                image_label = image_name[:-4]

                colmap_extrinsic = np.eye(4)
                Q_vec = np.array([qvec[1],qvec[2], qvec[3], qvec[0]])
                colmap_extrinsic[0:3, 0:3] = Rotation.from_quat(Q_vec).as_matrix()
                colmap_extrinsic[0:3, 3] = tvec
                colmap_extrinsic = np.linalg.inv(colmap_extrinsic)
                colmap_extrinsic[0:3, 3] -= translation
                
                xml_transform_string = ' '.join(map(str, colmap_extrinsic.reshape(-1)))
                # print(xml_transform_string)
                new_camera = ET.Element('camera')
                new_camera.set('id', str(image_id))
                new_camera.set('sensor_id', '0')
                new_camera.set('component_id', '0')
                new_camera.set('label', image_label)
                new_camera.text = '\n         '
                new_camera.tail = '\n      '

                transform_elem = ET.Element('transform')
                transform_elem.text = xml_transform_string
                transform_elem.tail = '\n      '

                new_camera.append(transform_elem)

                root.find('chunk/cameras').append(new_camera)

    return xml_tree

if __name__ == "__main__":
    args = argparser()

    colmap_file = args.colmap_path
    xml_file = args.xml_path
    output_path = args.output_path

    xml_tree = copy_xml_heads(xml_file=xml_file)
    xml_tree = copy_colmap_cams(xml_tree=xml_tree, colmap_path=colmap_file)
    xml_tree.write(output_path)

    