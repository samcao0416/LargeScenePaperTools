import argparse
import xml.etree.ElementTree as ET
import os
from copy import deepcopy

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-xml", "--xml_file", type=str, help="The camera xml file.")
    parser.add_argument("-out", "--output_folder", type=str, help="The output xml folder.")
    parser.add_argument("-n", "--num", type=int, default=100, help="The number of cameras in each xml file.")
    args = parser.parse_args()

    return args

def copy_xml_heads(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for e in root.findall('chunk/cameras')[0].findall("camera"):
        root.findall('chunk/cameras')[0].remove(e)

    return tree

def split_xml(xml_file, xml_tree, output_folder, num):

    root = ET.parse(xml_file).getroot()
    child = root.findall('chunk/cameras')[0][0]
    if child.tag == "camera":
        cnt = 0
        group_cnt = -1
        for e in root.findall('chunk/cameras')[0].findall("camera"):

            if e.find('transform') is None:
                continue

            if cnt == 0:
                tree = deepcopy(xml_tree)
                group_cnt += 1

                new_camera = ET.Element("camera")
                new_camera.set('id', e.get('id'))
                new_camera.set('sensor_id', e.get('sensor_id'))
                new_camera.set('component_id', e.get('component_id'))
                new_camera.set("label", e.get("label"))
                new_camera.text = '\n         '
                new_camera.tail = '\n      '

                transform_elem = ET.Element('transform')
                transform_elem.text = e.find("transform").text
                transform_elem.tail = '\n      '

                new_camera.append(transform_elem)
                tree.getroot().findall('chunk/cameras')[0].append(new_camera)

                cnt += 1


            elif cnt == (num-1):
                cnt = 0

                new_camera = ET.Element("camera")
                new_camera.set('id', e.get('id'))
                new_camera.set('sensor_id', e.get('sensor_id'))
                new_camera.set('component_id', e.get('component_id'))
                new_camera.set("label", e.get("label"))
                new_camera.text = '\n         '
                new_camera.tail = '\n      '

                transform_elem = ET.Element('transform')
                transform_elem.text = e.find("transform").text
                transform_elem.tail = '\n      '

                new_camera.append(transform_elem)
                tree.getroot().findall('chunk/cameras')[0].append(new_camera)

                output_path = os.path.join(output_folder, "split_%03d.xml" % group_cnt)
                tree.write(output_path)


            else:
                new_camera = ET.Element("camera")
                new_camera.set('id', e.get('id'))
                new_camera.set('sensor_id', e.get('sensor_id'))
                new_camera.set('component_id', e.get('component_id'))
                new_camera.set("label", e.get("label"))
                new_camera.text = '\n         '
                new_camera.tail = '\n      '

                transform_elem = ET.Element('transform')
                transform_elem.text = e.find("transform").text
                transform_elem.tail = '\n      '

                new_camera.append(transform_elem)
                tree.getroot().findall('chunk/cameras')[0].append(new_camera)


                cnt += 1

    elif child.tag == "group":
        for g in root.findall('chunk/cameras')[0].findall("group"):
            for e in g.findall("camera"):

                cnt = 0
                group_cnt = -1

                if cnt == 0:
                    tree = xml_tree.deepcopy()
                    group_cnt += 1

                    new_camera = ET.Element("camera")
                    new_camera.set('id', e.get('id'))
                    new_camera.set('sensor_id', e.get('sensor_id'))
                    new_camera.set('component_id', e.get('component_id'))
                    new_camera.set("label", e.get("label"))

                    transform_elem = ET.Element('transform')
                    transform_elem.text = e.find("transform").text

                    new_camera.append(transform_elem)
                    tree.getroot().findall('chunk/cameras')[0].append(new_camera)

                elif cnt == (num-1):
                    cnt = 0

                    new_camera = ET.Element("camera")
                    new_camera.set('id', e.get('id'))
                    new_camera.set('sensor_id', e.get('sensor_id'))
                    new_camera.set('component_id', e.get('component_id'))
                    new_camera.set("label", e.get("label"))

                    transform_elem = ET.Element('transform')
                    transform_elem.text = e.find("transform").text

                    new_camera.append(transform_elem)
                    tree.getroot().findall('chunk/cameras')[0].append(new_camera)

                    output_path = os.path.join(output_folder, "split_%03d.xml" % group_cnt)
                    tree.write(output_path)


                else:
                    new_camera = ET.Element("camera")
                    new_camera.set('id', e.get('id'))
                    new_camera.set('sensor_id', e.get('sensor_id'))
                    new_camera.set('component_id', e.get('component_id'))
                    new_camera.set("label", e.get("label"))

                    transform_elem = ET.Element('transform')
                    transform_elem.text = e.find("transform").text

                    new_camera.append(transform_elem)
                    tree.getroot().findall('chunk/cameras')[0].append(new_camera)

            cnt += 1

if __name__ == "__main__":
    args = argparser()

    xml_tree = copy_xml_heads(args.xml_file)

    os.makedirs(args.output_folder, exist_ok=True)
    split_xml(args.xml_file, xml_tree, args.output_folder, args.num)
    print("Done!")