import os
import sys
sys.path.append("../../")
sys.path.append("../")
sys.path.append("./")

from GAUSSIAN_Scripts.agi2colmap9 import Agi2Colmap, IndexRange

import numpy as np



K_xml_path = r"E:\SkdPaper\raw_data_bak\CS_Botanical\Garden_1\result\cam_insta.xml"

insta_class = Agi2Colmap(xml_path=r"E:\SkdPaper\raw_data_bak\CS_Botanical\Garden_1\result\poses.xml",
                         img_folder=r"E:\SkdPaper\raw_data_bak\CS_Botanical\Garden_1\result\images",
                         ply_path=r"E:\SkdPaper\raw_data_bak\CS_Botanical\Garden_1\result\result_3cm.ply",
                         mesh_path=r"E:\SkdPaper\raw_data_bak\CS_Botanical\Garden_1\mesh.ply",
                         point_pick_txt_path=r"E:\SkdPaper\CS_Botanical\Garden_1\test_agi8\picking_list.txt",
                         output_path=r"E:\SkdPaper\CS_Botanical\Garden_1\test_agi8",
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

insta_class.process_blocks.get_view_triangle_matrix(simplify=True) # NOTE: If all the job are done, you can just annotated this line
# insta_class.process_blocks.load_mesh() # NOTE: If the above line is annotated, you need to load the mesh by this line
insta_class.process_blocks.get_blocks(expand_threshold=0.3) # hyperparameter that control the expand ratio of quad bounding box
# NOTE: if you only want to proceed some of the block, use the following code
# insta_class.process_blocks.set_index(index_list=[0, 1, 3, 5])
insta_class.process_blocks.make_folders()
insta_class.process_blocks.crop_pcds()
insta_class.process_blocks.crop_mesh_get_matrix2()
# NOTE: if matrix2 has been calculated, use the following line:
# insta_class.process_blocks.load_matrix2()
# NOTE: if matrix3 or 4 or 5 is precaculated, change the "False" into "True" to save time.
# Priority: 5 > 4 > 3, which mean is matrix5 is loaded, then matrix3 and matrix4 will be ignored.
insta_class.process_blocks.select_images(threshold=0.01, load_matrix3=False, load_matrix4=False, load_matrix5=False) # hyperparameter that control the selection ratio of images. The ratio is the number of triangles in the cropped mesh the view can see to the total number of triangles in the cropped mesh. Perfer lower to higher.

# Or Alternatively, the above six lines can be replaced by the following line
# insta_class.process_blocks.run_default(quad_expand_threshold=0.3, img_selection_threshold=0.01)
# quad_expand_threshold and img_selection_threshold are hyperparameters
