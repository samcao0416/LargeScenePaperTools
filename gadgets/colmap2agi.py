import glob
import sys
from xml.dom import minidom
import pandas as pd
from scipy.spatial.transform import Rotation
import os
import numpy as np

def read_colmap_pose(pose_file):
    poses = []
    indexes = []
    image_names = []
    with open(pose_file, 'r') as f:
        while(True):
            line = f.readline()

            if not line:
                break

            line_ele = line.strip()

            if len(line_ele) > 0 and line[0] != "#":

                elems = line_ele.split()

                image_id = int(elems[0])

                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                try:
                    elems = f.readline().split()
                    xys = np.column_stack([tuple(map(float, elems[0::3])),
                                        tuple(map(float, elems[1::3]))])
                    point3D_ids = np.array(tuple(map(int, elems[2::3])))
                except:
                    xys = None
                    point3D_ids = None
                qvec = np.array([qvec[1], qvec[2], qvec[3], qvec[0]])
                pose = np.concatenate([tvec, qvec])
                poses.append(pose)
                indexes.append(image_id)
                image_names.append(image_name)

    return poses, indexes, image_names



def write_extern(filename, rotate, center, dom, Photogroup, idx, frame_count=0):
    Photo = dom.createElement('Photo')
    Photogroup.appendChild(Photo)

    ImagePath = dom.createElement('ImagePath')
    Photo.appendChild(ImagePath)
    path_text = dom.createTextNode(filename)
    ImagePath.appendChild(path_text)

    Idx = dom.createElement('Idx')
    Photo.appendChild(Idx)
    Idx.appendChild(dom.createTextNode(str(idx)))

    Frame_Count = dom.createElement('FrameCount')
    Photo.appendChild(Frame_Count)
    Frame_Count.appendChild(dom.createTextNode(str(frame_count)))

    Pose = dom.createElement('Pose')
    Photo.appendChild(Pose)
    Rotation = dom.createElement('Rotation')
    Pose.appendChild(Rotation)
    for i in range(3):
        for j in range(3):
            node = "M_" + str(i) + str(j)
            rotate_node = dom.createElement(node)
            rotate_node.appendChild(dom.createTextNode(str(rotate[j, i])))
            Rotation.appendChild(rotate_node)

    Center = dom.createElement('Center')
    Pose.appendChild(Center)
    cnter_label = ['x', 'y', 'z']
    for i in range(3):
        c_node = dom.createElement(cnter_label[i])
        c_node.appendChild(dom.createTextNode(str(center[i])))
        Center.appendChild(c_node)


def Init_photograph(dom, Photogroups, width_str='1920', height_str='1920'):
    Photogroup = dom.createElement('Photogroup')
    Photogroups.appendChild(Photogroup)

    dim = dom.createElement('ImageDimensions')
    Photogroup.appendChild(dim)
    Width = dom.createElement('Width')
    Height = dom.createElement('Height')
    Width_text = dom.createTextNode(width_str)
    Height_text = dom.createTextNode(height_str)
    dim.appendChild(Width)
    dim.appendChild(Height)
    Width.appendChild(Width_text)
    Height.appendChild(Height_text)

    return Photogroup

def output_extrinsic_xml(pose_file,image_folder,xml_file):
    
    # poses = pd.read_csv(pose_file,delimiter=' ',header=None).values
    poses, indexes, image_names = read_colmap_pose(pose_file)
    posix = image_names[0][-4:]
    image_files = glob.glob(image_folder+"/*.{}".format(posix))

    dom = minidom.Document()
    root_node = dom.createElement('BlocksExchange')
    dom.appendChild(root_node)
    uselessblock = dom.createElement('SpatialReferenceSystems')
    root_node.appendChild(uselessblock)
    uselessblock1 = dom.createElement('SRS')
    uselessblock.appendChild(uselessblock1)
    testid = dom.createElement('Id')
    testid_text = dom.createTextNode('1')
    testid.appendChild(testid_text)
    uselessblock1.appendChild(testid)
    testname = dom.createElement('Name')
    testname_text = dom.createTextNode('Local Coordinates (m)')
    testname.appendChild(testname_text)
    uselessblock1.appendChild(testname)
    testdef = dom.createElement('Definition')
    testdef_text = dom.createTextNode('LOCAL_CS[\"Local Coordinates (m)\",LOCAL_DATUM[\"Local Datum\",0],UNIT[\"metre\",1,AUTHORITY[\"EPSG\",\"9001\"]]]')
    testdef.appendChild(testdef_text)
    uselessblock1.appendChild(testdef)
    
    block = dom.createElement('Block')
    root_node.appendChild(block)
    Photogroups = dom.createElement('Photogroups')
    block.appendChild(Photogroups)


    Photogroup = Init_photograph(dom, Photogroups,'3072', '3072')

    photo_idx = 0

    for index, pose in enumerate(poses):
        extrinsics = np.eye(4)
        R = Rotation.as_matrix(Rotation.from_quat(pose[3:]))
        t = pose[0:3]
        extrinsics[:3, :3] = R
        extrinsics[:3, 3] = t
        extrinsics = np.linalg.inv(extrinsics)
        R = extrinsics[:3, :3]
        t = extrinsics[:3, 3]
        # timestamp = float(pose[0])
        # if len(poses) == len(image_files):
        #     image_file = image_files[index]
        # elif len(poses) > len(image_files):
        #     temp_file_name = "%06d.jpg" % (index)
        #     image_file = None
        #     for file_path in image_files:
        #         if os.path.basename(file_path) == temp_file_name:
        #             image_file = file_path
        #             break
        image_file = image_names[index]
        if image_file is not None:
            file_first_name = image_file.split('/')[-1].replace('images\\', '')

            write_extern(file_first_name, R, t, dom, Photogroup, photo_idx, 0)
            photo_idx += 1

    with open(xml_file,'w',encoding='UTF-8') as fh:
        dom.writexml(fh,indent='',addindent='\t',newl='\n',encoding='UTF-8')

if __name__ == "__main__":
    path = sys.argv[1]
    path = path.replace('\\', '/')
    if os.path.basename(path) != "images.txt" \
        or os.path.basename(path) != "images_train.txt" \
        or os.path.basename(path) != "images_test.txt" \
        or os.path.basename(path) != "images_all.txt":
        raise ValueError("Path should end up with images.txt or images_(train/test/all).txt")
    print(path)
    folder = os.path.dirname(os.path.dirname(os.path.dirname(path)))
    image_folder = folder + '/images'
    if os.path.basename(path) == "images.txt":
        xml_file = folder + '/camera_extern.xml'
    elif os.path.basename(path) == "images_test.txt":
        xml_file = folder + '/camera_extern_test.xml'
    elif os.path.basename(path) == "images_train.txt":
        xml_file = folder + '/camera_extern_train.xml'
    elif os.path.basename(path) == "images_all.txt":
        xml_file = folder + '/camera_extern_all.xml'
    output_extrinsic_xml(path,image_folder,xml_file)
