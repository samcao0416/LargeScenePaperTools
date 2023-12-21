# Version 2.0.1
# For Titan Camera System, folder structure:
#  - titan
#   - titan_0
#   - titan_1
#   - titan_2
#   - titan_4
#   - titan_5
#   - titan_6
#   - titan_7
#   - titan_8
#   - titan_9

import Metashape #type: ignore
import os
import argparse
import glob
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--img_folder", type = str, required = True, help = "path of titan folder")
parser.add_argument("--xml", type = str, default = None, help = "path of blocks exchange format path")
parser.add_argument("--workspace", type = str, default = None, help = "path of saving all the files")
args = parser.parse_args()

titan_folder = args.img_folder
if args.workspace is None:
    workspace_path = os.path.join(os.path.dirname(titan_folder), os.path.basename(titan_folder) + "_workspace")
    os.makedirs(workspace_path, exist_ok=True)
else:
    workspace_path = args.workspace
xml_path = args.xml

image_paths_list = [] # finally should be 8 * N, N = image_num

# check folders and images
for index in range(0, 10):
    if index == 3:
        pass
    else:
        folder_name = os.path.join(titan_folder, "titans_" + str(index))
        if not os.path.exists(folder_name):
            raise FileNotFoundError("Folder {} not found".format(folder_name))
        else:
            image_paths_list.append(sorted(glob.glob(os.path.join(folder_name, "*.jpg"))))
    try:
        image_folders_array = np.array(image_paths_list)
        for image_folder in image_paths_list:
            print("[ INFO ] ", image_folder, ": %d images" %(len(image_folder)))
    except:
        for image_folder in image_paths_list:
            print("[ ERROR ] ", image_folder, ": %d images" %(len(image_folder)))
        raise FileExistsError("image numbers do not match!")
    
images_list = np.transpose(image_folders_array[..., None], (1, 0, 2)).tolist()

# 1. 创建新的项目和chunk
doc = Metashape.Document()
chunk = doc.addChunk()

# 2. 导入图像
chunk.addPhotos(filenames = images_list, layout = Metashape.MultiplaneLayout)