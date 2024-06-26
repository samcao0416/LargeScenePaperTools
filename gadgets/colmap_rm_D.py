from scipy.spatial.transform import Rotation as R
import numpy as np
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-in", '--input_path', type=str, required=True)
    parser.add_argument('-out', '--output_path', type=str, required=True)
    return parser.parse_args()

def convert_colmap_to_vokf(qvec, tvec):
    qvec = np.array([qvec[1],qvec[2], qvec[3], qvec[0]])
    transforms = np.eye(4)
    transforms[:3, :3] = R.from_quat(qvec).as_matrix()
    transforms[:3, 3] = tvec
    transforms = np.linalg.inv(transforms)
    qvec = R.from_matrix(transforms[:3, :3]).as_quat()
    tvec = transforms[:3, 3]
    return qvec, tvec

if __name__ == "__main__":
    args = parse_args()

    in_txt_path = args.input_path
    out_txt_path = args.output_path

    os.makedirs(os.path.dirname(out_txt_path), exist_ok=True)

    with open(out_txt_path, "w+") as f_out:

        with open(in_txt_path, 'r') as f_in:
            image_id = 1
            while(True):
                line = f_in.readline()

                if not line:
                    break

                line_ele = line.strip()

                if len(line_ele) > 0 and line[0] != "#":

                    elems = line_ele.split()

                    
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

                    

                if image_name[-5] != "D":
                    image_name = image_name[:-4] + ".jpg"
                    output_line = f"{image_id } {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} {tvec[0]} {tvec[1]} {tvec[2]} 1 {image_name}\n No Content \n"
                    image_id += 1

                    f_out.write(output_line)
