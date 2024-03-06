import argparse
import xml.etree.ElementTree as ET
import numpy as np


def filter_cam(xml_file, view_list):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    child = root.findall('chunk/cameras')[0][0]
    if child.tag == "camera":
        for e in root.findall('chunk/cameras')[0].findall("camera"):
            label = e.get("label")
            if label not in view_list:
                root.findall('chunk/cameras')[0].remove(e)

    elif child.tag == "group":
        for g in root.findall('chunk/cameras')[0].findall("group"):
            for e in g.findall("camera"):
                label = e.get('label')
                if label not in view_list:
                    g.remove(e)

    else:
        raise ValueError("Unknown tag: %s" % child.tag)
    
    return tree

def get_view_list(colmap_file):
    
    image_names = []
    with open(colmap_file, "r") as fid:
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
                
                image_names.append(image_name)
    return image_names

def argparser():
    parser = argparse.ArgumentParser(description="Filter the camera xml file.")
    parser.add_argument("-xml", "--xml_file", type=str, help="The camera xml file.")
    parser.add_argument("-txt", "--colmap_file", type=str, help="The colmap txt file.")
    parser.add_argument("-out", "--output_xml", type=str, help="The output xml file.")
    return parser.parse_args()

if __name__ == "__main__":
    args = argparser()
    view_list = get_view_list(args.colmap_file)
    tree = filter_cam(args.xml_file, view_list)
    tree.write(args.output_xml)
    print("The filtered camera xml file has been saved to %s" % args.output_xml)