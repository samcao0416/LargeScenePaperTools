import os
import sys
sys.path.append("../../")
sys.path.append("../")
sys.path.append("./")

from GAUSSIAN_Scripts.agi2colmap8 import Agi2Colmap, IndexRange

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

insta_class.process_blocks.get_view_triangle_matrix()
insta_class.process_blocks.get_blocks(expand_threshold=0.3) # hyperparameter that control the expand ratio of quad bounding box
insta_class.process_blocks.make_folders()
insta_class.process_blocks.crop_pcds()
insta_class.process_blocks.crop_mesh_get_matrix2()
insta_class.process_blocks.select_images(threshold=0.01) # hyperparameter that control the selection ratio of images. The ratio is the number of triangles in the cropped mesh the view can see to the total number of triangles in the cropped mesh. Perfer lower to higher.

# Or Alternatively, the above six lines can be replaced by the following line
# insta_class.process_blocks.run_default(quad_expand_threshold=0.3, img_selection_threshold=0.01)
# quad_expand_threshold and img_selection_threshold are hyperparameters
