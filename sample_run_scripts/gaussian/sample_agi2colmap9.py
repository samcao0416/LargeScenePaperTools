import os
import sys
sys.path.append("../../")
sys.path.append("../")
sys.path.append("./")

from GAUSSIAN_Scripts.agi2colmap9 import Agi2Colmap, IndexRange

import numpy as np

raw_data_folder = r"E:\SkdPaper\raw_data_bak\demo_1215\SIG_ALL" # Change this line to your own path
output_folder = r"E:\SkdPaper\demo_1215\SIG_ALL" # Change this line to your own path

K_xml_path = os.path.join(raw_data_folder, "result", "cam_insta.xml")

insta_class = Agi2Colmap(xml_path=os.path.join(raw_data_folder,"result", "poses.xml"),
                         img_folder=os.path.join(raw_data_folder,"result", "images"),
                         ply_path=os.path.join(raw_data_folder,"result", "result_4cm.ply"),
                         mesh_path=os.path.join(raw_data_folder,"temp", "mesh.ply"),
                         point_pick_txt_path=os.path.join(output_folder,"picking_list.txt"),
                         output_path=output_folder,
                         data_type="insta")

insta_class.check_files.run()

insta_class.process_images.read_cams()

# Uncomment this line to select images with index range
# img_select_range = IndexRange(30,40,1)
# insta_class.process_images.select_image(img_select_range=img_select_range) 

insta_class.process_images.set_cam_param(K_xml=K_xml_path, new_width=734, new_height=734, fovX=np.pi / 2, method="fisheye2multiples", views = ["F", "L", "R", "U"])
insta_class.process_images.fisheye2multiples(write_mode=True)

# NOTE: if the all images have been split to multiples, you can load the cached pinhole cameras names and their corresponding poses by:
# insta_class.process_images.load_cam_pose_list_cache()

insta_class.process_blocks.get_view_triangle_matrix() # NOTE: If all the job are done, you can just annotated this line
# insta_class.process_blocks.load_mesh() # NOTE: If the above line is annotated, you need to load the mesh by this line

insta_class.process_blocks.get_blocks(expand_threshold=0.3) # hyperparameter that control the expand ratio of quad bounding box

# NOTE: if you only want to proceed some of the block, use the following code
# insta_class.process_blocks.set_index(index_list=[0, 1, 3, 5])

insta_class.process_blocks.make_folders()

# NOTE: if the pointclouds have already been cropped, you can annotated the following line to skip the step.
insta_class.process_blocks.crop_pcds()

insta_class.process_blocks.crop_mesh_get_matrix2(save_cropped_mesh=True)
# NOTE: if matrix2 has been calculated, use the following line:
# insta_class.process_blocks.load_matrix2()


# NOTE: if matrix3 or 4 or 5 is precaculated, change the "False" into "True" to save time.
# NOTE: Priority: 5 > 4 > 3, which mean is matrix5 is loaded, then matrix3 and matrix4 will be ignored.
# NOTE: set save_vokf=True to save the poses cloudcompare can read to visualize the selected images.
insta_class.process_blocks.select_images(threshold=0.8, load_matrix3=False, load_matrix4=False, load_matrix5=False, save_vokf=True) # hyperparameter that control the selection ratio of images. The ratio is the number of triangles seen in the block to the the number of triangles seen in the whole mesh.

# Or Alternatively, the above six lines can be replaced by the following line
# insta_class.process_blocks.run_default(quad_expand_threshold=0.3, img_selection_threshold=0.01)
# quad_expand_threshold and img_selection_threshold are hyperparameters


# NOTE: If you want to keep the cache files, you can annotated the following lines
insta_class.ClearCache()