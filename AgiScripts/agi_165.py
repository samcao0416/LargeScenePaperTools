# Version 1.6.5 
import Metashape #type: ignore
import os
import argparse

args = argparse.ArgumentParser()
parser = argparse.ArgumentParser()

parser.add_argument("--img_folder", type = str, required = True, help = "path of image folder")
parser.add_argument("--xml", type = str, default = None, help = "path of blocks exchange format path")
parser.add_argument("--workspace", type = str, default = None, help = "path of saving all the files")
args = parser.parse_args()

image_folder_path = args.img_folder
if args.workspace is None:
    workspace_path = os.path.join(os.path.dirname(image_folder_path), os.path.basename(image_folder_path) + "_workspace")
    os.makedirs(workspace_path, exist_ok=True)
else:
    workspace_path = args.workspace
xml_path = args.xml

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 创建新的项目和chunk
doc = Metashape.Document()
chunk = doc.addChunk()

# 2. 导入图像
  # 请将其替换为你的图片文件夹路径
image_list = os.listdir(image_folder_path)
image_paths = [os.path.join(image_folder_path, image) for image in image_list if image.endswith(('.jpg', '.png'))]
chunk.addPhotos(image_paths)

if xml_path is not None:
    # 3. 导入 blocks exchange 格式的标定文件
    chunk.importCameras(path = xml_path, format=Metashape.CamerasFormatBlocksExchange)

    # 4. 导出reference文件
    export_reference_path = os.path.join(workspace_path, "ref.txt")  # 请替换为你的导出路径
    chunk.exportReference(path=export_reference_path, format=Metashape.ReferenceFormatCSV, items=Metashape.ReferenceItemsCameras, columns = "nxyzabcuvwdef", delimiter=',')

    # 5. 在同一个chunk导入刚刚导出的reference文件
    chunk.importReference(path=export_reference_path, format=Metashape.ReferenceFormatCSV, items=Metashape.ReferenceItemsCameras, columns = "nxyzabcuvwdef", delimiter=',')

    # 6. 修改reference中xyz的误差为0.03m
    for marker in chunk.markers:
        marker.reference.accuracy = (0.03, 0.03, 0.03)

# 7. 修改相机模型，固定cx, cy = 0
# for camera in chunk.cameras:
#     if camera.calibration:
#         calibration = camera.calibration
#         calibration.cx = 0
#         calibration.cy = 0
#         calibration.fixed_parameters = [Metashape.Calibration.cx, Metashape.Calibration.cy]

# 8. 标定相机，reset当前以有的标定，以导入的reference为参考
    chunk.matchPhotos(downscale=1, generic_preselection=True, reference_preselection=True, keypoint_limit = 40000, tiepoint_limit = 9000)
    chunk.alignCameras(reset_alignment=True)

else:
    chunk.matchPhotos(downscale=1, generic_preselection=True, reference_preselection=False, keypoint_limit = 40000, tiepoint_limit = 9000)
    chunk.alignCameras(reset_alignment=True, adaptive_fitting = True)

# 9. build medium质量的点云
chunk.buildDepthMaps(downscale=1, filter_mode=Metashape.FilterMode.ModerateFiltering)
chunk.buildPointCloud(source_data = Metashape.DataSource.DepthMapsData)

# 10. 导出blocks exchange的xml标定和点云
export_calib_xml = os.path.join(workspace_path, "agi_blocks_exchange.xml") 
chunk.exportCameras(path=export_calib_xml, format=Metashape.CamerasFormatBlocksExchange, save_points = False)
export_dense_pcd = os.path.join(workspace_path, "agi_pcd.ply")  
chunk.exportPointCloud(path=export_dense_pcd, format=Metashape.PointCloudFormatPLY)

print("Script finished!")