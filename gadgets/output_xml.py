import glob
import argparse
import os
from xml.dom import minidom
import pandas as pd
from scipy.spatial.transform import Rotation


def get_file_dirname1(file_first_name):
    file_first_name = file_first_name.split('images/')[-1]
    return file_first_name

def get_file_dirname2(file_first_name):
    file_first_name = file_first_name.split('images\\')[-1]
    return file_first_name

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


def Init_photograph(dom, Photogroups, width_str='2000', height_str='2000'):
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
    posix = 'jpg'
    poses = pd.read_csv(pose_file,delimiter=' ',header=None).values
    image_files = glob.glob(image_folder+"/*.{}".format(posix))

    dom = minidom.Document()
    blocksexchange_node = dom.createElement('BlocksExchange')
    blocksexchange_node_version = dom.createAttribute('version')
    blocksexchange_node_version.nodeValue = '3.2'
    blocksexchange_node.setAttributeNode(blocksexchange_node_version)
    dom.appendChild(blocksexchange_node)
    spatial = dom.createElement('SpatialReferenceSystems')
    # block = dom.createElement('Block')
    blocksexchange_node.appendChild(spatial)
    SRS = dom.createElement('SRS')
    spatial.appendChild(SRS)
    id = dom.createElement('Id')
    id_text = dom.createTextNode('1')
    id.appendChild(id_text)
    SRS.appendChild(id)
    name = dom.createElement('Name')
    name_text = dom.createTextNode('Local Coordinates (m)')
    name.appendChild(name_text)
    SRS.appendChild(name)
    definition = dom.createElement('Definition')
    definition_text = dom.createTextNode('LOCAL_CS[\"Local Coordinates (m)\",LOCAL_DATUM[\"Local Datum\",0],UNIT[\"metre\",1,AUTHORITY[\"EPSG\",\"9001\"]]]')
    definition.appendChild(definition_text)
    SRS.appendChild(definition)
    Photogroups = dom.createElement('Photogroups')
    # block.appendChild(Photogroups)

    block = dom.createElement('Block')
    blocksexchange_node.appendChild(block)
    name = dom.createElement('Name')
    name_text = dom.createTextNode('Chunk 1')
    name.appendChild(name_text)
    block.appendChild(name)
    description = dom.createElement('Description')
    description_text = dom.createTextNode('Result of aerotriangulation of Chunk 1 (2022-Oct-01 15:15:48)')
    description.appendChild(description_text)
    block.appendChild(description)
    SRSid = dom.createElement('SRSId')
    SRSid_text = dom.createTextNode('1')
    SRSid.appendChild(SRSid_text)
    block.appendChild(SRSid)
    Photogroups = dom.createElement('Photogroups')
    block.appendChild(Photogroups)


    Photogroup = Init_photograph(dom, Photogroups,'2000', '2000')

    photo_idx = 0

    file_first_name = image_files[0].split('/')[-1]
    if os.path.dirname(file_first_name.split('images\\')[-1]) == "":
        get_file_dirname = get_file_dirname1
    else:
        get_file_dirname = get_file_dirname2

    for pose in poses:
        R = Rotation.as_matrix(Rotation.from_quat(pose[4:]))
        t = pose[1:4]
        timestamp = float(pose[0])
        for image_file in image_files:
            image_timestamp = float(image_file.split('\\')[-1].replace('.{}'.format(posix),''))

            if abs(image_timestamp-timestamp)<0.001:
                file_first_name = image_file.split('/')[-1]
                file_first_name = get_file_dirname(file_first_name)
                file_first_name = file_first_name.split('.jpg')[0]
                file_first_name = file_first_name + ".jpg"
                write_extern(file_first_name, R, t, dom, Photogroup, photo_idx, 0)
                photo_idx += 1
                break

    with open(xml_file,'w',encoding='UTF-8') as fh:
        dom.writexml(fh,indent='',addindent='\t',newl='\n',encoding='UTF-8')

if __name__ == "__main__":
    # folder_list = ["Keyuan", "SCA", "SEM", "SIST", "SLST", "SPST"]
    # for folder in folder_list:
    #     posefile = "./" + folder + '/result/vo_kf.txt'
    #     image_folder = "./" + folder + '/result/images'

    #     xml_file ="./"+  folder  + '/result/camera.xml'
    #     print(posefile)
    #     output_extrinsic_xml(posefile,image_folder,xml_file)

    parser = argparse.ArgumentParser()
    parser.add_argument("-txt", "--vo_kf_path", type=str)
    parser.add_argument("-img", "--img_folder", type=str)
    parser.add_argument("-out", "--output_path", type=str, default=None)
    args = parser.parse_args()

    posefile = args.vo_kf_path
    image_folder = args.img_folder

    if args.output_path is None:
        xml_file = os.path.dirname(args.img_folder) + "./camera.xml"
    else:
        xml_file = args.output_path
    
    output_extrinsic_xml(posefile,image_folder,xml_file)